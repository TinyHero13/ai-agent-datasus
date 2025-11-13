# Databricks notebook source
# MAGIC %md
# MAGIC # Agente SRAG

# COMMAND ----------

# MAGIC %pip install -qU langchain-anthropic langchain-community
# MAGIC %pip install -U langchain langchain-core langchain-community
# MAGIC %pip install -qU sqlalchemy databricks-sql-connector databricks-sqlalchemy
# MAGIC %pip install -qU sqlalchemy databricks-sql-connector databricks-sqlalchemy
# MAGIC %pip install -U langchain langchain-core langchain-community
# MAGIC %pip install langchain-tavily
# MAGIC dbutils.library.restartPython()
# MAGIC %restart_python

# COMMAND ----------

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from datetime import datetime
import json
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
import base64
from IPython.display import HTML
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.secrets.get(scope = "srag_agent", key = "DATABRICKS_TOKEN")
SERVER_HOSTNAME = dbutils.secrets.get(scope = "srag_agent", key = "SERVER_HOSTNAME")
HTTP_PATH = dbutils.secrets.get(scope = "srag_agent", key = "HTTP_PATH")

ANTHROPIC_API_KEY = dbutils.secrets.get(scope = "srag_agent", key = "ANTHROPIC_API_KEY")
TAVILY_API_KEY = dbutils.secrets.get(scope = "srag_agent", key = "TAVILY_API_KEY")

TABLE_NAME = "srag"

catalog = "srag_datasus"
schema = "gold"

engine = create_engine(
    url=f"databricks://token:{DATABRICKS_TOKEN}@{SERVER_HOSTNAME}?"
        f"http_path={HTTP_PATH}&catalog={catalog}&schema={schema}"
)

db = SQLDatabase(
    engine=engine,
    include_tables=["srag"],
    sample_rows_in_table_info=3
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração do Agente SQL e Tavily para pesquisas

# COMMAND ----------

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=2000
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

tools += [TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
    country="Brazil"
)]

print("Tools disponíveis:")
for tool in tools:
    print(f"{tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. System prompt

# COMMAND ----------

system_prompt = f"""Você é um analista epidemiológico especializado em SRAG.

**TABELA:** {TABLE_NAME}

**COLUNAS PRINCIPAIS:**
- data_sintomas (DATE)
- evolucao (STRING): 'Óbito', 'Cura', etc
- uti (STRING): 'Sim', 'Não', '1', '2'
- vacina_covid (STRING)
- idade (INT)
- uf (STRING)

**REGRAS:**
1. SEMPRE use DATE_SUB(CURRENT_DATE(), N) para datas
2. SEMPRE use UPPER(CAST(coluna AS STRING)) para comparações de texto
3. SEMPRE use NULLIF para evitar divisão por zero
4. LIMITE queries a 100 registros
5. Para óbito: UPPER(CAST(evolucao AS STRING)) LIKE '%OBITO%'
6. Para UTI: UPPER(CAST(uti AS STRING)) IN ('SIM', '1')
7. Para vacinação: vacina_covid IS NOT NULL AND UPPER(CAST(vacina_covid AS STRING)) NOT IN ('NAO', '2')

**OBJETIVO:**
Gerar relatório com 4 métricas principais + análise temporal + contexto.

**FERRAMENTAS DISPONÍVEIS:**
1. SQL Database Tools - Para consultar dados históricos no banco
2. Tavily Search - Para buscar notícias e contexto atual sobre SRAG no Brasil

**QUANDO USAR TAVILY:**
- SEMPRE busque notícias recentes sobre SRAG após calcular as métricas
- Use termos como: "SRAG Brasil", "síndrome respiratória aguda grave", "surto respiratório"
- Busque informações sobre: surtos atuais, alertas sanitários, políticas públicas
- Use as notícias para CONTEXTUALIZAR e EXPLICAR as métricas calculadas

**FLUXO DE TRABALHO:**
1. Execute queries SQL para calcular as 4 métricas principais
2. Gere os dados para os 2 gráficos (diário e mensal)
3. Use Tavily para buscar notícias atuais sobre SRAG no Brasil
4. Analise correlações entre métricas e notícias
5. Gere relatório final integrando dados + contexto jornalístico
"""

# COMMAND ----------

agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Execução

# COMMAND ----------

prompt = """Gere um relatório epidemiológico SRAG completo seguindo estes passos:

1. **Taxa de Aumento de Casos**
   - Compare casos da última semana vs semana anterior
   - Use: data_sintomas >= DATE_SUB(CURRENT_DATE(), 14)

2. **Taxa de Mortalidade (últimos 30 dias)**
   - Total de casos
   - Total de óbitos (UPPER(CAST(evolucao AS STRING)) LIKE '%OBITO%')
   - Taxa percentual

3. **Taxa de Ocupação UTI (últimos 30 dias)**
   - Total de casos
   - Internações UTI (UPPER(CAST(uti AS STRING)) IN ('SIM', '1'))
   - Taxa percentual

4. **Taxa de Vacinação (últimos 30 dias)**
   - Total de casos
   - Vacinados (vacina_covid IS NOT NULL AND UPPER(CAST(vacina_covid AS STRING)) NOT IN ('NAO', '2'))
   - Taxa percentual

5. **Tendência Diária** - Casos por dia (últimos 10 dias)

6. **Tendência Mensal** - Casos por mês (últimos 6 meses)

7. **CONTEXTO ATUAL (OBRIGATÓRIO)**
   - Use a ferramenta Tavily Search para buscar notícias recentes sobre:
     * "SRAG Brasil últimas semanas"
     * "surto respiratório Brasil"
     * "síndrome respiratória aguda grave notícias"
   - Identifique: alertas sanitários, surtos regionais, campanhas de vacinação
   - Correlacione as notícias com as métricas calculadas

Formate a resposta assim:

# RELATÓRIO SRAG

## MÉTRICAS PRINCIPAIS
[mostre valores reais de cada métrica]

## ANÁLISE TEMPORAL
[descreva tendências observadas nos gráficos]

## CONTEXTO ATUAL E NOTÍCIAS
[apresente as notícias encontradas e como elas se relacionam com as métricas]

## CONCLUSÕES
[análise final integrando dados históricos + contexto atual das notícias]

IMPORTANTE: Você DEVE usar o Tavily Search para buscar notícias antes de gerar as conclusões.
"""

for step in agent.stream(
    {"messages": [{"role": "user", "content": prompt}]},
    stream_mode="values"
):
    if "messages" in step:
        step["messages"][-1].pretty_print()

# COMMAND ----------

def gerar_relatorio_srag_com_guardrails(max_tentativas=3):
    """Executa o agente com guardrails e auto-correção"""
    global prompt
    
    log_file = f"agent_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    for tentativa in range(1, max_tentativas + 1):
        result = None
        tools_usadas = []
        
        for step in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values"
        ):
            if "messages" in step:
                msg = step["messages"][-1]
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_usadas.append({
                            "timestamp": datetime.now().isoformat(),
                            "tool": tool_call.get("name", "unknown"),
                            "args": tool_call.get("args", {})
                        })
                
                result = step["messages"][-1].content
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Tentativa {tentativa} - {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(json.dumps(tools_usadas, indent=2, ensure_ascii=False))
            f.write("\n")
        
        print(f"Ferramentas utilizadas: {len(tools_usadas)}")
        tavily_usado = False
        for uso in tools_usadas:
            print(f"  - {uso['timestamp']}: {uso['tool']}")
            if 'tavily' in uso['tool'].lower():
                tavily_usado = True
        
        validacao = validar_relatorio(result)
        
        if validacao["valido"]:
            print("Relatório aprovado")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Status: Aprovado na tentativa {tentativa}\n")
            
            return result, tools_usadas
        else:
            print(f"Problemas encontrados na tentativa {tentativa}:")
            for aviso in validacao["avisos"]:
                print(f"  {aviso}")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Problemas encontrados")
                for aviso in validacao["avisos"]:
                    f.write(f"  - {aviso}\n")
            
            if tentativa < max_tentativas:
                feedback = gerar_feedback_correcao(validacao["avisos"], result, tavily_usado)
                print("Enviando correções para o agente")
                prompt = feedback
            else:
                print("Numero maximo de tentativas atingido")
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nStatus: Maximo de tentativas atingido\n")
                
                return result, tools_usadas
    
    return result, tools_usadas

# COMMAND ----------

def validar_relatorio(relatorio_texto):
    """Guardrails para validar se o relatório atende aos requisitos"""
    
    avisos = []
    texto_lower = relatorio_texto.lower()
    
    metricas_checklist = {
        "taxa de aumento": ["taxa de aumento", "variação", "crescimento", "redução de casos"],
        "mortalidade": ["mortalidade", "óbitos", "taxa de óbito"],
        "UTI": ["uti", "unidade de terapia intensiva", "internações uti", "taxa de uti"],
        "vacinação": ["vacinação", "vacinados", "imunização", "cobertura vacinal"]
    }
    
    for metrica_nome, variacoes in metricas_checklist.items():
        if not any(var in texto_lower for var in variacoes):
            avisos.append(f"MÉTRICA_AUSENTE: {metrica_nome}")
    
    secoes_checklist = {
        "MÉTRICAS PRINCIPAIS": ["métricas principais", "métricas"],
        "ANÁLISE TEMPORAL": ["análise temporal", "tendência", "temporal"],
        "CONTEXTO ATUAL": ["contexto atual", "notícias", "cenário"],
        "CONCLUSÕES": ["conclusões", "conclusão", "recomendação"]
    }
    
    for secao_nome, variacoes in secoes_checklist.items():
        if not any(var in texto_lower for var in variacoes):
            avisos.append(f"SEÇÃO_AUSENTE: {secao_nome}")
    
    elementos_contexto = ["notícia", "surto", "alerta", "fiocruz", "ministerio", "sanitário"]
    if not any(elem in texto_lower for elem in elementos_contexto):
        avisos.append("CONTEXTO_AUSENTE: Relatório não inclui contexto atual")
    
    if len(relatorio_texto) < 1000:
        avisos.append("RELATÓRIO_INCOMPLETO: Conteúdo muito curto")
    
    return {
        "valido": len(avisos) == 0,
        "avisos": avisos
    }

# COMMAND ----------

def gerar_feedback_correcao(avisos, relatorio_anterior, tavily_usado):
    """Gera prompt de correção baseado nos problemas encontrados"""
    
    feedback_parts = [
        "ATENÇÃO: O relatório anterior apresentou problemas. Corrija-o seguindo estas instruções:\n"
    ]
    
    metricas_faltando = [a.split(": ")[1] for a in avisos if "MÉTRICA_AUSENTE" in a]
    if metricas_faltando:
        feedback_parts.append(f"""
### MÉTRICAS FALTANDO:
{', '.join(metricas_faltando)}

AÇÃO REQUERIDA: Execute queries SQL para calcular estas métricas e inclua-as na seção "MÉTRICAS PRINCIPAIS" com valores numéricos específicos (percentuais, totais, etc).
""")
    
    secoes_faltando = [a.split(": ")[1] for a in avisos if "SEÇÃO_AUSENTE" in a]
    if secoes_faltando:
        feedback_parts.append(f"""
### SEÇÕES FALTANDO:
{', '.join(secoes_faltando)}

AÇÃO REQUERIDA: Adicione estas seções com conteúdo substantivo ao relatório.
""")
    
    if any("CONTEXTO_AUSENTE" in a for a in avisos):
        feedback_parts.append(f"""
###  CRÍTICO - CONTEXTO ATUAL AUSENTE:

AÇÃO OBRIGATÓRIA: 
1. Use a ferramenta Tavily Search AGORA com estas queries:
   - "SRAG Brasil últimas semanas"
   - "surto respiratório Brasil {datetime.now().year}"
   - "síndrome respiratória aguda grave notícias"

2. Adicione uma seção "## CONTEXTO ATUAL E NOTÍCIAS" contendo:
   - Resumo das notícias encontradas
   - Alertas sanitários identificados
   - Correlação entre notícias e métricas calculadas

Tavily foi usado? {'SIM' if tavily_usado else 'NÃO - VOCÊ DEVE USÁ-LO AGORA'}
""")
    
    if any("ANÁLISE_TEMPORAL" in a for a in avisos):
        feedback_parts.append(f"""
### ANÁLISE TEMPORAL INSUFICIENTE:

AÇÃO REQUERIDA: Adicione à seção "ANÁLISE TEMPORAL":
1. Gráfico/dados de tendência diária (últimos 30 dias)
2. Gráfico/dados de tendência mensal (últimos 12 meses)
3. Interpretação das tendências (crescimento, estabilização, queda)
""")
    
    if any("INCOMPLETO" in a for a in avisos):
        feedback_parts.append(f"""
### RELATÓRIO INCOMPLETO:

O relatório está muito curto ({len(relatorio_anterior)} caracteres). 
AÇÃO REQUERIDA: Expanda todas as seções com análises detalhadas.
""")
    
    feedback_parts.append(f"""
---
### RELATÓRIO ANTERIOR (para referência):
{relatorio_anterior[:1000]}... [truncado]

---
### INSTRUÇÕES FINAIS:
- Mantenha o que estava BOM no relatório anterior
- Corrija APENAS os problemas listados acima
- Gere o relatório COMPLETO novamente com todas as seções
- Use as ferramentas necessárias (SQL para dados, Tavily para notícias)
""")
    
    return "\n".join(feedback_parts)

# COMMAND ----------

def salvar_relatorio_pdf(texto, nome_arquivo="relatorio_srag.pdf"):
    """Gera e salva o relatório SRAG em formato PDF profissional"""
    
    if texto.startswith("Agora vou gerar"):
        linhas = texto.split("\n")
        texto = "\n".join(linhas[1:])
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=50, 
        leftMargin=50,
        topMargin=50, 
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    titulo_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a4d7a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    secao_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5f8d'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subsecao_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#3d7ab8'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    lista_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        fontName='Helvetica'
    )

    story = []
    
    linhas = texto.split("\n")
    
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        
        if not linha:
            story.append(Spacer(1, 6))
            i += 1
            continue
        
        linha_limpa = re.sub(r'\*\*(.*?)\*\*', r'\1', linha)
        linha_limpa = re.sub(r'\*(.*?)\*', r'\1', linha_limpa)
        
        if '|' in linha and linha.count('|') >= 2:
            tabela_linhas = []
            while i < len(linhas) and '|' in linhas[i]:
                tabela_linhas.append(linhas[i].strip())
                i += 1
            
            tabela_dados = []
            for tl in tabela_linhas:
                if re.match(r'^\|[\s\-]+\|', tl):
                    continue
                
                colunas = [c.strip() for c in tl.split('|') if c.strip()]
                if colunas:
                    tabela_dados.append(colunas)
            
            if tabela_dados:
                t = Table(tabela_dados)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5f8d')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
            continue
        
        if linha.startswith("# ") and not linha.startswith("## "):
            titulo = linha_limpa.replace("# ", "")
            story.append(Paragraph(titulo, titulo_style))
            story.append(Spacer(1, 12))
        
        elif linha.startswith("## "):
            secao = linha_limpa.replace("## ", "")
            story.append(Spacer(1, 8))
            story.append(Paragraph(secao, secao_style))
            story.append(Spacer(1, 6))
        
        elif linha.startswith("### "):
            subsecao = linha_limpa.replace("### ", "")
            story.append(Paragraph(subsecao, subsecao_style))
            story.append(Spacer(1, 4))
        
        elif linha.startswith("- ") or linha.startswith("* ") or linha.startswith("• "):
            item = linha_limpa.lstrip("- *•").strip()
            story.append(Paragraph(f"• {item}", lista_style))
        
        elif re.match(r'^\d+\.', linha):
            story.append(Paragraph(linha_limpa, lista_style))
        
        elif linha.startswith("```"):
            pass
        
        else:
            story.append(Paragraph(linha_limpa, normal_style))
        
        story.append(Spacer(1, 4))
        i += 1

    doc.build(story)
    buffer.seek(0)

    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a download="{nome_arquivo}" href="data:application/pdf;base64,{b64}" target="_blank">Baixar Relatório PDF</a>'
    display(HTML(href))
    
    print(f"PDF gerado: {nome_arquivo}")
relatorio_final, audit_log = gerar_relatorio_srag_com_guardrails(max_tentativas=3)

salvar_relatorio_pdf(relatorio_final)

