# TCC

# Trabalho de Conclusão de Curso – Engenharia Ambiental
**Modelagem de intensidade de metano (CH₄) no upstream offshore brasileiro de óleo e gás**

Este projeto implementa o pipeline analítico do TCC que investiga a intensidade de metano associada ao flaring no upstream offshore brasileiro, utilizando **exclusivamente dados públicos da ANP** e conectando os resultados às exigências do **Regulamento (UE) 2024/1787** sobre emissões de metano.

O trabalho combina:
- **ETL em Power Query (Excel/Power BI)** para consolidar e padronizar bases ANP 2020–2022 e 2023–2025 em m³/mês;  
- **Análises em Python** para cálculo de intensidade de flaring, séries temporais, curvas de Pareto e relações estatísticas entre variáveis operacionais.

Mais do que propor um “modelo de mercado”, o projeto busca mostrar, de forma transparente e reprodutível, **como dados públicos podem ser usados para estimar e projetar a intensidade de flaring offshore**, identificar ativos mais críticos (Top 10) e discutir riscos de conformidade regulatória.

## Módulos principais

- **Consolidação e padronização de dados (Power Query / M)**  
  - Combinação das bases ANP 2020–2022 e 2023–2025;  
  - Conversão de Mm³/dia e m³/dia para m³/mês;  
  - Cálculo de `% Reinjeção` e `Intensidade Flaring (%)`;  
  - Junção com base de instalações para trazer `Idade do Ativo` e `Atendimento Campo`;  
  - Geração da base consolidada `Dados_Finais_V3.xlsx`.

- **Análises em Python (script `VF_DiogoMiranda_CodeTCC`)**  
  - Leitura e preparação da base consolidada (`load_and_prepare`);  
  - Foco em instalações **offshore**, com filtro de qualidade adicional (remoção de meses com produção e queima simultaneamente nulas);  
  - **Séries temporais** de intensidade agregada e média móvel de 12 meses (2020–2025);  
  - **Curva de Pareto Top 10** instalações por gás queimado e participação no total offshore;  
  - **Correlação** entre `% Reinjeção` e `Intensidade Flaring (%)` (Pearson e Spearman);  
  - **Regressão potência em escala log–log** entre gás produzido e gás queimado, com estimativa de expoente e R²;  
  - **Validação 80/20 por instalação**, usando a média histórica de intensidade como baseline de previsão.

Os resultados evidenciam tanto o potencial quanto as limitações de modelos simples baseados apenas em dados operacionais agregados: embora permitam inferências relevantes sobre tendência e concentração de emissões, apresentam restrições importantes para previsão fina de intensidade por instalação, o que dialoga com o debate atual sobre MRV, transparência e risco regulatório.

## Dependências

Este projeto foi desenvolvido e testado em ambiente Python com as seguintes bibliotecas:

- Python ≥ 3.10  
- `pandas` ≥ 2.0  
- `numpy` ≥ 1.24  
- `matplotlib` ≥ 3.7  
- `openpyxl` ≥ 3.1  

Para instalar as dependências mínimas necessárias em um ambiente virtual, basta executar:

```bash
pip install "pandas>=2.0" "numpy>=1.24" "matplotlib>=3.7" "openpyxl>=3.1"
