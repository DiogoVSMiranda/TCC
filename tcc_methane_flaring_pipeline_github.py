# -*- coding: utf-8 -*-
"""
TCC — Intensidade de metano associada ao flaring no upstream brasileiro (ANP).

Este script lê o dataset consolidado (data_ANP.xlsx ou .csv) e gera, em lote:
- Série temporal de intensidade (média móvel 12m, ponderada por gás produzido);
- Distribuições (boxplots) por tipo de instalação;
- Reinjecção vs intensidade (scatter), incluindo diagnóstico de outliers;
- Correlação (Pearson e Spearman) e regressão log–log (GQ vs GP);
- Projeções lineares exploratórias até 2030 em diferentes janelas históricas;
- Top 10 por gás queimado (absoluto) e por intensidade média (normalizado);
- Validação temporal 80/20 por instalação (MAE/MAPE) + histograma.

Estrutura recomendada para publicar no GitHub:
  repo/
    data/data_ANP.xlsx
    tcc_methane_flaring_pipeline.py

Execução:
  python tcc_methane_flaring_pipeline.py

Saídas:
  ./out_tcc/figs (figuras) e arquivos auxiliares (.csv/.xlsx/.json)
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# --------------------------- Parâmetros Globais ---------------------------
DATA_PATH = os.path.join("data", "data_ANP.xlsx")  # default repo path
OUT_DIR = "./out_tcc"
SHEET_NAME = 0
DATE_MIN, DATE_MAX = "2020-01-01", "2025-03-01"
TOPK_INST = 10
PALETA_JSON = os.path.join(OUT_DIR, 'campo_colors.json')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (11, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

TOP10_USE_LOG10_X = False
TOP10_X_LIMIT = 4.0
TOP10_SCALE_UNIT = 1e11  # ajuste opcional do eixo X no Top 10 (None para remover)
ONLY_MARITIMO_FOR_TYPE = True

# Ajustes solicitados
MIN_MESES_TOP10_INT = 6
XMAX_INTENSIDADE_TOP10 = 180

# ------------------------------- Utilitários ------------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def_repl = {
    "Estação":"E.","Estações":"E.","Est.":"E.","Estação/":"E./",
    "Coletora":"Colet.","Ponto":"Pto.","Petróleo":"Petr.","Gás Natural":"Gás Nat.",
    "Armazenamento":"Armz.","Tratamento":"Trat.","Central":"Ctral.","Fazenda":"Fz."
}

def shorten_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    for k,v in def_repl.items():
        s = s.replace(k,v)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:35] + ("…" if len(s) > 35 else "")

def _parse_pt_percent_maybe_fraction(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        x = (s.astype(str).str.strip()
                .str.replace('%','',regex=False)
                .str.replace('.','',regex=False)
                .str.replace(',', '.', regex=False))
        s = pd.to_numeric(x, errors='coerce')
    else:
        s = pd.to_numeric(s, errors='coerce')
    fin = s[np.isfinite(s)]
    if len(fin)>0 and np.nanpercentile(fin,90)<=1.2:
        s = s*100.0
    return s

def as_percent_0_100(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors='coerce').astype(float)
    finite = x[np.isfinite(x)]
    if len(finite)==0:
        return x
    p90 = np.nanpercentile(finite,90)
    if p90<=1.2:
        return x*100.0
    elif p90>=120:
        return x/100.0
    else:
        return x

def parse_pct_onepass(series: pd.Series) -> pd.Series:
    """Converte strings para número e normaliza para 0–100 em UMA etapa.
    - Remove '%'; troca ',' por '.'; mantém '.'
    - Heurística de escala: se p90<=1.2 => *100; se p90>=120 => /100; senão mantém
    - Clampa 0–100
    """
    x = (series.astype(str).str.strip()
                .str.replace('%','',regex=False)
                .str.replace(',', '.', regex=False))
    num = pd.to_numeric(x, errors='coerce')
    finite = num[pd.notna(num) & ~np.isinf(num)]
    if len(finite)>0:
        p90 = float(np.nanpercentile(finite, 90))
        if p90 <= 1.2:
            num = num * 100.0
        elif p90 >= 120.0:
            num = num / 100.0
    return num.clip(lower=0, upper=100)

def drop_both_zero(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df['gas_queimado'].fillna(0)==0)&(df['gas_produzido'].fillna(0)==0))].copy()

# Paleta persistida por grupo (aceita force_distinct para compatibilidade)
def color_map_from_series(series: pd.Series, default_key='Não informado', force_distinct: bool=False) -> dict:
    ensure_dir(os.path.dirname(PALETA_JSON))
    keys = series.fillna(default_key).replace({'':default_key}).astype(str).unique().tolist()
    keys_sorted = [c for c in sorted(keys) if c!=default_key]
    n = max(1,len(keys_sorted))
    if n<=20:
        base = list(plt.cm.tab20.colors)[:n]
    else:
        try:
            import seaborn as sns
            base = sns.color_palette('husl',n)
        except Exception:
            base = (list(plt.cm.tab20.colors)*(n//20+1))[:n]
    cmap = {k:tuple(col) for k,col in zip(keys_sorted,base)}
    cmap[default_key]=(0.6,0.6,0.6)
    try:
        with open(PALETA_JSON,'w',encoding='utf-8') as f:
            json.dump(cmap,f,ensure_ascii=False,indent=2)
    except Exception:
        pass
    return cmap

# -------------------------- Leitura & Preparação -------------------------
def load_and_prepare(path_xlsx: str, sheet=SHEET_NAME) -> pd.DataFrame:
    df = None
    if os.path.exists(path_xlsx):
        try:
            if path_xlsx.lower().endswith(('.xls','.xlsx')):
                df = pd.read_excel(path_xlsx, sheet_name=sheet, engine='openpyxl')
            else:
                raise Exception('not excel')
        except Exception:
            df = None
    if df is None:
        csv_guess = path_xlsx if path_xlsx.lower().endswith('.csv') else os.path.splitext(path_xlsx)[0]+'.csv'
        if os.path.exists(csv_guess):
            try:
                df = pd.read_csv(csv_guess, sep=';')
            except Exception:
                df = pd.read_csv(csv_guess)
        else:
            raise FileNotFoundError(f"Não encontrei {path_xlsx} nem {csv_guess}")

    rename_map = {
        '\ufeffPeríodo':'periodo','Período':'periodo','Periodo':'periodo',
        'Nome da Instalação':'instalacao_nome','Tipo de Instalação':'instalacao_tipo',
        'Ambiente Instalação':'instalacao_ambiente','Campo':'campo',
        'Bacia':'bacia','Bacia Sedimentar':'bacia_sedimentar',
        'Gás Queimado':'gas_queimado','Gás Produzido':'gas_produzido','Gás Injetado':'gas_injetado',
        'Gás Lift':'gas_lift','Gás Consumido':'gas_consumo','Gás Queimado Ventilado':'gas_ventilado',
        'Intensidade Flaring (%)':'intensidade_flaring_pct','% Reinjeção':'reinjecao_pct',
        'Idade do Ativo':'idade_ativo','Atendimento Campo':'atendimento_campo'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # Datas
    if 'periodo' not in df.columns:
        raise KeyError('Coluna "Período/periodo" não encontrada após renomeação.')
    df['periodo'] = pd.to_datetime(df['periodo'], dayfirst=True, errors='coerce')
    df['Mes'] = df['periodo'].dt.to_period('M').dt.to_timestamp()
    df = df[(df['periodo']>=pd.to_datetime(DATE_MIN)) & (df['periodo']<=pd.to_datetime(DATE_MAX))].copy()

    # Numéricos
    for c in ['gas_queimado','gas_produzido','gas_injetado','gas_lift','gas_consumo','gas_ventilado','idade_ativo']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Percentuais – agora a base já está em 0–100, não precisa escalar nem clapar
    if 'reinjecao_pct' in df.columns:
        df['reinjecao_pct'] = pd.to_numeric(df['reinjecao_pct'], errors='coerce')
    if 'intensidade_flaring_pct' in df.columns:
        df['intensidade_flaring_pct'] = pd.to_numeric(df['intensidade_flaring_pct'], errors='coerce')

    # Ambiente marítimo
    if 'instalacao_ambiente' in df.columns:
        amb = df['instalacao_ambiente'].astype(str)
        mask_mar = (
            amb.str.contains('mar', case=False, na=False)
            | amb.str.contains('marít', case=False, na=False)
            | amb.str.contains('offshore', case=False, na=False)
        )
        df['is_mar'] = mask_mar if mask_mar.any() else False
    else:
        df['is_mar'] = False

    # Campo / Atendimento Campo
    if 'campo' in df.columns:
        df['campo'] = df['campo'].fillna('Não informado').astype(str)
    else:
        df['campo'] = pd.Series(['Não informado']*len(df), index=df.index).astype(str)

    if 'atendimento_campo' in df.columns:
        df['atendimento_campo'] = df['atendimento_campo'].fillna('').astype(str)
    else:
        df['atendimento_campo'] = pd.Series(['']*len(df), index=df.index).astype(str)

    df['atendimento_campo_clean'] = (
        df['atendimento_campo'].astype(str)
            .str.replace(r'\s*\(.*?\)', '', regex=True)
            .str.strip()
            .replace({'': 'Não informado'})
    )

    # Somente não-negativos
    if 'gas_produzido' in df.columns:
        df = df[df['gas_produzido']>=0]
    if 'gas_queimado' in df.columns:
        df = df[df['gas_queimado']>=0]

    # Se não veio intensidade calculada, calcula
    if ('intensidade_flaring_pct' not in df.columns) or df['intensidade_flaring_pct'].isna().all():
        if 'gas_produzido' not in df.columns or 'gas_queimado' not in df.columns:
            df['intensidade_flaring_pct'] = np.nan
        else:
            denom = df['gas_produzido']
            frac = np.where((denom>0)&pd.notna(denom), (df['gas_queimado']/denom), np.nan)
            df['intensidade_flaring_pct'] = as_percent_0_100(frac)
    return df

# -------------------------- Série de Intensidade --------------------------

def serie_intensidade(df: pd.DataFrame) -> pd.DataFrame:
    if 'Mes' not in df.columns:
        raise ValueError("DataFrame precisa ter 'Mes'. Rode load_and_prepare antes.")
    agg = (df.groupby('Mes', as_index=False)
             .agg(gq=('gas_queimado','sum'), gp=('gas_produzido','sum'))
             .sort_values('Mes'))
    agg = agg[agg['gp']>0].copy()
    agg['int_pct'] = (agg['gq']/agg['gp'])*100.0
    agg['int_pct_ma12'] = agg['int_pct'].rolling(window=12, min_periods=1).mean()
    return agg[['Mes','int_pct','int_pct_ma12']]

# ------------------------------- 1) Série -------------------------------

def fig_intensidade_offshore(df,out_dir):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    agg = serie_intensidade(d)
    plt.figure(figsize=(12,6))
    plt.plot(agg['Mes'], agg['int_pct_ma12'], marker='o', linewidth=1.8, label='MM 12m (ponderada)')
    plt.xlabel('Mês'); plt.ylabel('Intensidade de Flaring (%)')
    plt.grid(True, ls=':'); plt.legend(loc='upper left'); plt.tight_layout()
    p = os.path.join(out_dir, 'fig_intensidade_offshore_serie.png')
    plt.savefig(p, dpi=200); plt.close(); return p

def fig_intensidade_offshore_sem_queima(df,out_dir):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    cond_sem = ((d['gas_queimado'].fillna(0)==0) & (d['gas_produzido'].fillna(0)>0))
    n_sem = int(cond_sem.sum())
    d2 = d[~cond_sem].copy()
    agg = serie_intensidade(d2)
    plt.figure(figsize=(12,6))
    plt.plot(agg['Mes'], agg['int_pct_ma12'], marker='o', linewidth=1.8, color='#ff7f0e', label='MM 12m (ponderada)')
    plt.xlabel('Mês'); plt.ylabel('Intensidade de Flaring (%)')
    plt.grid(True, ls=':'); plt.legend(loc='upper left')
    plt.annotate(f"Meses com produção>0 e queima=0 excluídos: {n_sem}", xy=(0.5,-0.15), xycoords='axes fraction', ha='center', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#444', alpha=0.9))
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig_intensidade_offshore_sem_queima.png')
    plt.savefig(p, dpi=200); plt.close()
    info = pd.DataFrame([{'criterio':'producao>0_e_queima=0','n_excluido': n_sem}])
    info.to_csv(os.path.join(out_dir, 'fig_intensidade_offshore_sem_queima_excluidos.csv'), index=False)
    return p, n_sem

# ----------------- 2) Boxplots por tipo + Resumo estatístico --------------

def fig_intensidade_por_tipo(df, out_dir):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if ONLY_MARITIMO_FOR_TYPE and 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    if 'instalacao_tipo' not in d.columns:
        return None
    d['instalacao_tipo'] = d['instalacao_tipo'].fillna('Sem tipo').astype(str)
    ban = {'sonda perfuracao', 'sonda perfuração', 'navio sonda'}
    d = d[~d['instalacao_tipo'].str.lower().isin(ban)].copy()
    d = d[d['gas_produzido'] > 0].copy()
    d['intensidade_pct'] = (d['gas_queimado'] / d['gas_produzido']) * 100.0
    d = d[np.isfinite(d['intensidade_pct'])].copy()
    if d.empty:
        return None
    in_range = (d['intensidade_pct'] >= 0) & (d['intensidade_pct'] <= 100)
    n_out = int((~in_range).sum())
    d_ok = d[in_range].copy()
    if d_ok.empty:
        return None
    resumo_tipo = (
        d_ok.groupby('instalacao_tipo')['intensidade_pct']
            .agg(mediana='median',
                 q1=lambda s: np.nanpercentile(s, 25),
                 q3=lambda s: np.nanpercentile(s, 75),
                 n='count')
            .reset_index()
    )
    resumo_tipo.to_csv(os.path.join(out_dir, 'intensidade_por_tipo_resumo.csv'), index=False)
    tipos_ordenados = (resumo_tipo.sort_values('mediana', ascending=False)['instalacao_tipo'].tolist())
    data_bp = [d_ok.loc[d_ok['instalacao_tipo'] == t, 'intensidade_pct'] for t in tipos_ordenados]
    plt.figure(figsize=(12, max(6, 0.45 * len(tipos_ordenados) + 1.5)))
    bp = plt.boxplot(
        data_bp,
        vert=False,
        tick_labels=tipos_ordenados,
        showfliers=True,
        patch_artist=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor(plt.cm.Blues(0.6))
        patch.set_alpha(0.7)
    plt.title('Distribuição da Intensidade de Flaring por Tipo de Instalação (offshore)')
    plt.xlabel('Intensidade de Flaring (%)')
    plt.ylabel('Tipo de instalação')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    plt.grid(True, ls=':', alpha=0.5)
    if n_out > 0:
        plt.annotate(
            f"Fora de 0–100%: {n_out}",
            xy=(0.98, 0.98),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#444', alpha=0.9)
        )
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig_intensidade_por_tipo.png')
    plt.savefig(p, dpi=200)
    plt.close()
    return p

# -------------------------- 3) Projeções (baseline) ----------------------

def figs_projecoes_janelas(df, out_dir, y_lim=(0.0,10.0), y_pad=0.0, clip_0_100=False):
    """Projeções lineares (baseline) da intensidade agregada offshore com Y fixo 0–10%."""
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    agg = serie_intensidade(d).copy()
    if agg.empty:
        def _mk_empty(fname, msg):
            pth = os.path.join(out_dir, fname)
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, msg, ha='center', va='center')
            plt.axis('off'); plt.savefig(pth, dpi=200); plt.close(); return pth
        msg = 'Sem dados suficientes para projeção.'
        return (_mk_empty('fig_proj_2020_2023.png', msg),
                _mk_empty('fig_proj_2023_2025.png', msg),
                _mk_empty('fig_proj_2020_2025.png', msg))
    y_min, y_max = float(y_lim[0]), float(y_lim[1])
    if y_max - y_min < 1e-6:
        y_min, y_max = (0.0, 10.0)
    if clip_0_100:
        y_min = max(0.0, y_min); y_max = min(100.0, y_max)
    y_lim = (y_min, y_max)

    def _proj(agg_local, start, end, fname):
        sub = agg_local[(agg_local['Mes']>=pd.Timestamp(start)) & (agg_local['Mes']<=pd.Timestamp(end))].copy()
        plt.figure(figsize=(12,6))
        if sub.empty:
            plt.text(0.5, 0.5, f'Sem dados suficientes na janela {start[:4]}–{end[:4]}', ha='center', va='center')
            plt.axis('off'); p = os.path.join(out_dir, fname); plt.savefig(p, dpi=200); plt.close(); return p
        plt.plot(sub['Mes'], sub['int_pct'], marker='o', linewidth=1.8, label='Histórico (ponderado)')
        if len(sub) >= 2:
            sub = sub.copy()
            sub['mi'] = sub['Mes'].dt.year*12 + sub['Mes'].dt.month
            t0 = sub['mi'].min(); sub['t'] = (sub['mi'] - t0).astype(float)
            a, b = np.polyfit(sub['t'], sub['int_pct'], 1)
            t_last = float(sub['t'].iloc[-1]); y_last = float(sub['int_pct'].iloc[-1])
            b_anchor = y_last - a*t_last
            last_m = pd.Period(sub['Mes'].max(), freq='M'); end_proj = pd.Period('2030-12', freq='M')
            proj = pd.DataFrame({'Mes': pd.period_range(last_m, end_proj, freq='M').to_timestamp()})
            proj['mi'] = proj['Mes'].dt.year*12 + proj['Mes'].dt.month
            proj['t'] = (proj['mi'] - t0).astype(float)
            proj['yhat_pct'] = a*proj['t'] + b_anchor
            proj_f = proj[proj['Mes'] >= sub['Mes'].max()].copy()
            plt.plot(proj_f['Mes'], proj_f['yhat_pct'], '--', linewidth=1.8, label='Projeção linear (ancorada)')
        plt.ylim(y_lim[0], y_lim[1])
        plt.title(f'Projeção da Intensidade Offshore até 2030 — Janela {start[:4]}–{end[:4]}')
        plt.xlabel('Mês'); plt.ylabel('Intensidade de Flaring (%)')
        plt.grid(True, ls=':'); plt.legend()
        plt.annotate('Ajuste linear exploratório (sem sazonalidade/variáveis explicativas)',
                     xy=(0.5,-0.12), xycoords='axes fraction', ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#444', alpha=0.9))
        plt.tight_layout(); p = os.path.join(out_dir, fname); plt.savefig(p, dpi=200); plt.close(); return p

    p1 = _proj(agg, '2020-01-01', '2023-12-31', 'fig_proj_2020_2023.png')
    p2 = _proj(agg, '2023-01-01', '2025-12-31', 'fig_proj_2023_2025.png')
    p3 = _proj(agg, '2020-01-01', '2025-12-31', 'fig_proj_2020_2025.png')
    return p1, p2, p3

# ---------------------- 4) Top10 por Gás Queimado ------------------------

def sci_label(val, mantissa_dec=2):
    """Formata número em notação científica compacta (mant × 10^exp)."""
    import math
    if not np.isfinite(val) or val == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / (10**exp)
    return f"{mant:.{mantissa_dec}f}x10^{exp}"

def fig_top10_gq_por_campo(df,out_dir):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    group_series_raw = d['atendimento_campo_clean'] if 'atendimento_campo_clean' in d.columns else d['campo']
    group_series = (group_series_raw.astype(str)
                        .str.replace(r'\s*\(.*?\)', '', regex=True)
                        .str.strip()
                        .replace({'':'Não informado'}))
    cmap = color_map_from_series(group_series, force_distinct=True)
    total_gq = d['gas_queimado'].sum()
    g = (d.groupby('instalacao_nome', as_index=False)['gas_queimado'].sum()
           .sort_values('gas_queimado', ascending=False).head(TOPK_INST))
    if 'atendimento_campo_clean' in d.columns:
        mode_map = d.groupby('instalacao_nome')['atendimento_campo_clean'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else 'Não informado')
    else:
        mode_map = d.groupby('instalacao_nome')['campo'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else 'Não informado')
    mode_map = mode_map.apply(lambda s: re.sub(r'\s*\(.*?\)', '', str(s)).strip() or 'Não informado')
    g = g.merge(mode_map.rename('campo_cor'), on='instalacao_nome', how='left')
    g['pct_total'] = (g['gas_queimado']/total_gq*100.0) if total_gq>0 else 0.0
    g = g.sort_values('gas_queimado', ascending=True)
    top10_share = float(g['pct_total'].sum())
    pd.DataFrame([{'top10_share_pct_do_total_offshore': top10_share}]).to_csv(os.path.join(out_dir, 'top10_gas_queimado_share.csv'), index=False)
    labels = [shorten_name(s) for s in g['instalacao_nome']]
    x_vals = g['gas_queimado'].astype(float).values.copy()
    x_label = 'Gás Queimado (m³)'
    if TOP10_USE_LOG10_X:
        x_vals = np.log10(np.maximum(x_vals, 1.0)); x_label = 'log10(Gás Queimado [m³])'
    elif TOP10_SCALE_UNIT is not None and TOP10_SCALE_UNIT>0:
        x_vals = x_vals / TOP10_SCALE_UNIT
        x_label = f"Gás Queimado (×10^{int(np.log10(TOP10_SCALE_UNIT))} m³)"
    colors = [cmap.get(c, (0.6,0.6,0.6)) for c in g['campo_cor']]
    h = max(6, 0.55*len(g)+1.5)
    plt.figure(figsize=(12,h))
    bars = plt.barh(labels, x_vals, color=colors, edgecolor='black', linewidth=0.6)
    if TOP10_USE_LOG10_X or (TOP10_SCALE_UNIT is not None):
        plt.xlim(0, TOP10_X_LIMIT)
    plt.title('Top 10 Instalações por Gás Queimado — valores ( ) = % do total')
    plt.xlabel(x_label); plt.ylabel('Instalação (abreviado)')
    uniq = pd.Series(g['campo_cor']).fillna('Não informado').unique()
    cats = sorted(set(uniq), key=lambda x: (x=='Não informado', str(x)))
    if 'Não informado' not in cats:
        cats = ['Não informado'] + cats
    handles = [Patch(facecolor=cmap.get(c,(0.6,0.6,0.6)), edgecolor='black', label=c) for c in cats]
    legend_title = 'Atendimento Campo' if 'atendimento_campo_clean' in d.columns else 'Campo'
    plt.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02,1), loc='upper left')
    x_max = float(max(x_vals) if len(x_vals)>0 else 1.0)
    for i, (b, val_raw, pct) in enumerate(zip(bars, g['gas_queimado'], g['pct_total'])):
        txt = f"{sci_label(val_raw)} ({pct:.1f}%)"; y = b.get_y() + b.get_height()/2; cur = b.get_width()
        inside = (not (TOP10_USE_LOG10_X or TOP10_SCALE_UNIT)) and (cur >= 0.65*x_max)
        if inside:
            plt.annotate(txt, (cur*0.98, y), ha='right', va='center', fontsize=9, color='white',
                         bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.4))
        else:
            voff = (-4 if i%2==0 else 4)
            plt.annotate(txt, (cur, y), xytext=(6, voff), textcoords='offset points', ha='left', va='center', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#333', alpha=0.8), clip_on=False)
    if TOP10_USE_LOG10_X:
        ticks = np.arange(0, TOP10_X_LIMIT+0.1, 1.0)
        plt.xticks(ticks, [rf"$10^{{{int(t)}}}$" for t in ticks])
    plt.tight_layout(); p = os.path.join(out_dir, 'fig_top10_gas_queimado_por_campo.png')
    plt.savefig(p, dpi=200); plt.close(); return p

# ---------- 5) Top10 por intensidade média (normalizado/robusto) ---------

def fig_top10_intensidade_media_por_instalacao(df, out_dir, k=TOPK_INST):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    d = d[d['gas_produzido']>0].copy()
    d['intensidade_pct'] = (d['gas_queimado'] / d['gas_produzido']) * 100.0
    # Categoria por instalação: Atendimento Campo (limpo) ou Campo
    if 'atendimento_campo_clean' in d.columns:
        mode_map = d.groupby('instalacao_nome')['atendimento_campo_clean'] \
            .agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else 'Não informado')
        legenda_titulo = 'Atendimento Campo'
    else:
        mode_map = d.groupby('instalacao_nome')['campo'] \
            .agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else 'Não informado')
        legenda_titulo = 'Campo'
    mode_map = mode_map.apply(lambda s: re.sub(r'\s*\(.*?\)', '', str(s)).strip() or 'Não informado')
    g = (d.groupby('instalacao_nome', as_index=False)
           .agg(int_media=('intensidade_pct','mean'), n_meses=('intensidade_pct','count')))
    g = g[g['n_meses'] >= MIN_MESES_TOP10_INT]
    g = g.merge(mode_map.rename('cat_cor'), on='instalacao_nome', how='left')
    g = g.sort_values('int_media', ascending=False).head(k).copy()
    cmap = color_map_from_series(g['cat_cor'], force_distinct=True)
    cats = sorted(set(g['cat_cor'].fillna('Não informado')), key=lambda x: (x=='Não informado', str(x)))
    if 'Não informado' not in cats:
        cats = ['Não informado'] + cats
    colors = [cmap.get(c, (0.6,0.6,0.6)) for c in g['cat_cor']]
    labels = [shorten_name(s) for s in g['instalacao_nome']]
    plt.figure(figsize=(12, max(6, 0.55*len(g)+1.5)))
    bars = plt.barh(labels, g['int_media'], color=colors, edgecolor='black')
    plt.gca().invert_yaxis()
    for y, (val, n) in enumerate(zip(g['int_media'], g['n_meses'])):
        plt.annotate(f"{val:.2f}% (n={int(n)})", xy=(min(val, XMAX_INTENSIDADE_TOP10*0.98), y), xytext=(6, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#333', alpha=0.8))
    plt.xlabel('Intensidade média de flaring (%)'); plt.ylabel('Instalação (abreviado)')
    plt.title('Top 10 por intensidade média de flaring (normalizado) — ' + f'n≥{MIN_MESES_TOP10_INT} — cores por ' + legenda_titulo)
    ax = plt.gca(); ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    plt.xlim(0, XMAX_INTENSIDADE_TOP10)
    handles = [Patch(facecolor=cmap.get(c,(0.6,0.6,0.6)), edgecolor='black', label=c) for c in cats]
    plt.legend(handles=handles, title=legenda_titulo, bbox_to_anchor=(1.02,1), loc='upper left')
    plt.grid(True, ls=':', alpha=0.5); plt.subplots_adjust(right=0.80); plt.tight_layout()
    p = os.path.join(out_dir, 'fig_top10_intensidade_media_por_instalacao.png')
    plt.savefig(p, dpi=200); plt.close(); return p

# ---------------- 6) Reinjecao vs Intensidade + outliers ------------------

def fig_reinjecao_vs_intensidade(df,out_dir, out_xlsx_name='reinjecao_intensidade_maiores_que_100.xlsx'):
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    cols = ['Mes','instalacao_nome','reinjecao_pct','intensidade_flaring_pct']
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    d = d[cols].copy()
    d['reinjecao_pct'] = pd.to_numeric(d['reinjecao_pct'], errors='coerce')
    d['intensidade_flaring_pct'] = pd.to_numeric(d['intensidade_flaring_pct'], errors='coerce')
    d = d.dropna(subset=['reinjecao_pct','intensidade_flaring_pct'])

    # Diagnóstico da reinjeção
    reinj = d['reinjecao_pct'].astype(float)
    vc = reinj.round(1).value_counts().sort_index()
    vc.rename('freq').to_csv(os.path.join(out_dir, 'diagnostico_reinjecao_valores.csv'), header=True)
    plt.figure(figsize=(8,4))
    plt.hist(reinj, bins=np.arange(0, 101, 2), color='#1f77b4', edgecolor='black', alpha=0.75)
    plt.xlabel('Reinjeção (%)'); plt.ylabel('Frequência'); plt.title('Distribuição de reinjeção (%)')
    plt.grid(True, ls=':', alpha=0.5); plt.subplots_adjust(right=0.80); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diagnostico_reinjecao_hist.png'), dpi=200); plt.close()

    # Outliers >100 para Excel
    big = d[(d['reinjecao_pct']>100) & (d['intensidade_flaring_pct']>100)].copy() \
            .sort_values(['reinjecao_pct','intensidade_flaring_pct'], ascending=False)
    xlsx_path = os.path.join(out_dir, out_xlsx_name)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        resumo = pd.DataFrame({'total_outliers':[len(big)],
                               'outliers_reinjecao_>100':[int((d['reinjecao_pct']>100).sum())],
                               'outliers_intensidade_>100':[int((d['intensidade_flaring_pct']>100).sum())]})
        resumo.to_excel(writer, index=False, sheet_name='Resumo')
        inst = (big.assign(flag=1).groupby('instalacao_nome', as_index=False)['flag']
                    .sum().rename(columns={'flag':'ocorrencias_>100'})) if len(big)>0 else pd.DataFrame(columns=['instalacao_nome','ocorrencias_>100'])
        inst.to_excel(writer, index=False, sheet_name='Instalacoes')

    in_0_100 = d['reinjecao_pct'].between(0,100) & d['intensidade_flaring_pct'].between(0,100)
    n_out = int((~in_0_100).sum())
    plt.figure(figsize=(8,8))
    dd = d[in_0_100]
    if len(dd)>0:
        eps = np.random.default_rng(42).normal(0, 0.6, size=len(dd))
        xj = np.clip(dd['reinjecao_pct'].values + eps, 0, 100)
        plt.scatter(xj, dd['intensidade_flaring_pct'], s=32, alpha=0.7, c='#1f77b4', edgecolor='black', linewidths=0.25)
        plt.xlim(0,100); plt.ylim(0,100)
        plt.xticks(range(0,101,20)); plt.yticks(range(0,101,20))
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        eps = np.random.default_rng(42).normal(0, 0.6, size=len(d))
        xj = np.clip(d['reinjecao_pct'].values + eps, 0, 100)
        plt.scatter(xj, d['intensidade_flaring_pct'], s=32, alpha=0.7, c='#1f77b4', edgecolor='black', linewidths=0.25)
    if n_out>0:
        plt.annotate(f"Fora de 0–100%: {n_out}", xy=(0.98,0.98), xycoords='axes fraction', ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#444', alpha=0.9))
    plt.title('Reinjeção (%) vs Intensidade de Flaring (%) — foco 0–100%')
    plt.xlabel('Reinjeção (%)'); plt.ylabel('Intensidade de Flaring (%)')
    plt.grid(True, which='both', ls=':'); plt.tight_layout()
    p_scatter = os.path.join(out_dir, 'fig_reinjecao_vs_intensidade.png')
    plt.savefig(p_scatter, dpi=200); plt.close(); return p_scatter, None, n_out, xlsx_path

# ------------------ 7) Correlação (Pearson/Spearman) ---------------------

def resumo_correlacao_reinj_intensidade(df, out_dir):
    """Gera PNG e CSV com correlação (Pearson/Spearman) entre reinjeção (%) e intensidade (%)."""
    ensure_dir(out_dir)
    d = drop_both_zero(df).copy()
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    for c in ['reinjecao_pct','intensidade_flaring_pct']:
        if c not in d.columns:
            d[c] = np.nan
 
    d['reinjecao_pct'] = pd.to_numeric(d['reinjecao_pct'], errors='coerce')
    d['intensidade_flaring_pct'] = pd.to_numeric(d['intensidade_flaring_pct'], errors='coerce')
    d = d.dropna(subset=['reinjecao_pct','intensidade_flaring_pct'])
    d['reinjecao_pct'].round(1).value_counts().sort_index().to_csv(os.path.join(out_dir, 'diagnostico_reinjecao_valores_corr.csv'), header=['freq'])
    x = d['reinjecao_pct'].astype(float).values
    y = d['intensidade_flaring_pct'].astype(float).values
    def _pearson(a,b):
        a = a - np.nanmean(a); b = b - np.nanmean(b)
        sa, sb = np.nanstd(a), np.nanstd(b)
        return float(np.nanmean(a*b) / (sa*sb)) if (sa>0 and sb>0) else float('nan')
    def _spearman(a,b):
        ra = pd.Series(a).rank(method='average').values
        rb = pd.Series(b).rank(method='average').values
        return _pearson(ra, rb)
    r_p = _pearson(x,y); r_s = _spearman(x,y)
    cor_path = os.path.join(out_dir, 'correlacao_reinj_intensidade_resumo.csv')
    pd.DataFrame([{'pearson_r': r_p, 'spearman_rho': r_s, 'n': int(len(x))}]).to_csv(cor_path, index=False)
    plt.figure(figsize=(8,8))
    eps = np.random.default_rng(42).normal(0, 0.6, size=len(x))
    xj = np.clip(x + eps, 0, 100)
    plt.scatter(xj, y, s=32, alpha=0.7, c='#1f77b4', edgecolor='black', linewidths=0.25)
    plt.xlim(0,100); plt.ylim(0,100)
    plt.xticks(range(0,101,20)); plt.yticks(range(0,101,20))
    plt.gca().set_aspect('equal', adjustable='box')
    txt_ann = f"Pearson r={r_p:.2f}\nSpearman rho={r_s:.2f}\nn={len(x)}"
    plt.title('Reinjeção (%) vs Intensidade de Flaring (%) — correlação')
    plt.xlabel('Reinjeção (%)'); plt.ylabel('Intensidade de Flaring (%)')
    plt.grid(True, ls=':')
    plt.annotate(txt_ann, xy=(0.5,-0.12), xycoords='axes fraction', ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#444', alpha=0.9))
    p = os.path.join(out_dir, 'correlacao_reinj_intensidade.png')
    plt.tight_layout(); plt.savefig(p, dpi=200); plt.close()
    return p, cor_path

# ---------------- 8) Log–log GQ vs GP + R² do ajuste potência -------------

def fig_log_gq_vs_gp_simple(df,out_dir):
    """Gráfico log–log de Gás Produzido vs Gás Queimado com ajuste potência e R²."""
    ensure_dir(out_dir)
    d = drop_both_zero(df)
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    d = d[['gas_queimado','gas_produzido']].dropna()
    d = d[(d['gas_queimado']>0) & (d['gas_produzido']>0)].copy()
    plt.figure(figsize=(8.5,7))
    if len(d)==0:
        plt.text(0.5,0.5,'Sem dados positivos para log-log', ha='center', va='center')
    else:
        x = d['gas_produzido'].values; y = d['gas_queimado'].values
        plt.scatter(x,y,s=12,alpha=0.6,c='#1f77b4',label='Observações')
        plt.xscale('log'); plt.yscale('log')
        lx, ly = np.log10(x), np.log10(y)
        b, a = np.polyfit(lx, ly, 1)
        yhat = a + b*lx
        ss_res = float(np.sum((ly - yhat)**2))
        ss_tot = float(np.sum((ly - np.mean(ly))**2))
        r2 = float('nan') if ss_tot<=0 else float(1 - ss_res/ss_tot)
        xline = np.linspace(lx.min(), lx.max(), 200)
        plt.plot(10**xline, 10**(a + b*xline), color='#1f77b4', lw=1.5, label='Ajuste potência')
        A = 10**a
        plt.annotate(f"y ≈ {A:.2e} · x^{b:.2f}\nR²={r2:.2f}", xy=(0.62,0.08), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1f77b4', alpha=0.9))
        plt.legend(loc='upper left')
    plt.xlabel('Gás Produzido (m³/mês) — log'); plt.ylabel('Gás Queimado (m³/mês) — log')
    plt.title('Gás Produzido vs Gás Queimado (log–log)'); plt.grid(True, which='both', ls=':')
    plt.tight_layout(); p = os.path.join(out_dir, 'fig_log_gq_vs_gp.png')
    plt.savefig(p, dpi=200); plt.close(); return p

# ------------ 9) Quem aumentou / reduziu (variação Δ entre janelas) ------




def _plot_bar_delta(df_top, title, fname, out_dir, color='#1f77b4', xmax=50.0, side='auto'):
    labels = [shorten_name(s) for s in df_top.index]
    plt.figure(figsize=(12.5, max(6, 0.55*len(df_top)+1.5)))
    ax = plt.gca()
    bars = plt.barh(labels, df_top['delta'], color=color, edgecolor='black')
    ax.invert_yaxis()

    # Detect side if auto
    if side == 'auto':
        dmin = float(np.nanmin(df_top['delta'])) if len(df_top)>0 else 0.0
        dmax = float(np.nanmax(df_top['delta'])) if len(df_top)>0 else 0.0
        side = 'pos' if dmin >= 0 else ('neg' if dmax <= 0 else 'mix')

    # Configure limits and spine so que x=0 fique colado aos rótulos do eixo Y
    if side == 'pos':
        ax.set_xlim(0, float(xmax))
        # Colar a spine esquerda no x=0 e manter rótulos à esquerda
        ax.spines['left'].set_position(('data', 0))
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', pad=2)
        plt.subplots_adjust(left=0.15, right=0.88)
    elif side == 'neg':
        ax.set_xlim(-float(xmax), 0)
        # Para negativos, levar a spine DIREITA para x=0 e deslocar rótulos para a direita
        ax.spines['right'].set_position(('data', 0))
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', pad=2)
        plt.subplots_adjust(left=0.10, right=0.88)
    else:  # mix
        span = float(xmax)
        ax.set_xlim(-span, span)
        # Em mix, manter spine esquerda padrão e apenas desenhar a linha em x=0
        ax.spines['left'].set_visible(True)
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', pad=2)
        plt.subplots_adjust(left=0.15, right=0.88)

    # Anotações
    for y, (dlt, a, b) in enumerate(zip(df_top['delta'], df_top['int_A'], df_top['int_B'])):
        if dlt >= 0:
            x_ann = min(dlt, ax.get_xlim()[1]*0.98); ha = 'left'; off = 6
        else:
            x_ann = max(dlt, ax.get_xlim()[0]*0.98); ha = 'right'; off = -6
        txt = 'Δ={:+.2f} pp\nA={:.2f}% → B={:.2f}%'.format(dlt, a, b)
        plt.annotate(txt, xy=(x_ann, y), xytext=(off, 0), textcoords='offset points',
                     ha=ha, va='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#333', alpha=0.8))

    # Linha em x=0
    ax.axvline(0, color='#666', lw=1)
    plt.xlabel('Δ Intensidade (pontos percentuais)')
    plt.ylabel('Instalação (abreviado)')
    plt.title(title)
    plt.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, fname)
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(); return p

def figs_quem_aumentou_e_reduziu(df, out_dir, k=TOPK_INST,
                                 jan1_a='2020-01-01', jan2_a='2022-12-31',
                                 jan1_b='2023-01-01', jan2_b='2025-12-31'):
    ensure_dir(out_dir)
    d = drop_both_zero(df).copy()
    if 'is_mar' in d.columns and d['is_mar'].any():
        d = d[d['is_mar']].copy()
    d = d[d['gas_produzido']>0].copy()
    d['intensidade_pct'] = (d['gas_queimado']/d['gas_produzido'])*100.0
    d = d[(d['intensidade_pct']>=0) & (d['intensidade_pct']<=100)].copy()
    A = d[(d['periodo']>=pd.to_datetime(jan1_a)) & (d['periodo']<=pd.to_datetime(jan2_a))].copy()
    B = d[(d['periodo']>=pd.to_datetime(jan1_b)) & (d['periodo']<=pd.to_datetime(jan2_b))].copy()
    if A.empty or B.empty or 'instalacao_nome' not in d.columns:
        p_inc = os.path.join(out_dir, 'fig_quem_aumentou_placeholder.png')
        p_red = os.path.join(out_dir, 'fig_quem_reduziu_placeholder.png')
        for pth, txt in [(p_inc,'Sem dados suficientes (A)'), (p_red,'Sem dados suficientes (B)')]:
            plt.figure(figsize=(8,4)); plt.text(0.5,0.5,txt,ha='center',va='center')
            plt.axis('off'); plt.savefig(pth, dpi=200); plt.close()
        eff_csv = os.path.join(out_dir, 'quem_aumentou_eficiente_resumo.csv')
        pd.DataFrame(columns=['instalacao_nome','int_A','int_B','delta','n_A','n_B']).to_csv(eff_csv, index=False)
        return p_inc, p_red, eff_csv
    gA = (A.groupby('instalacao_nome').agg(int_A=('intensidade_pct','mean'), n_A=('intensidade_pct','count')))
    gB = (B.groupby('instalacao_nome').agg(int_B=('intensidade_pct','mean'), n_B=('intensidade_pct','count')))
    g = (gA.join(gB, how='inner').dropna().copy())
    if g.empty:
        p_inc = os.path.join(out_dir, 'fig_quem_aumentou_placeholder.png')
        p_red = os.path.join(out_dir, 'fig_quem_reduziu_placeholder.png')
        for pth, txt in [(p_inc,'Sem interseção de instalações A∩B'), (p_red,'Sem interseção de instalações A∩B')]:
            plt.figure(figsize=(8,4)); plt.text(0.5,0.5,txt,ha='center',va='center')
            plt.axis('off'); plt.savefig(pth, dpi=200); plt.close()
        eff_csv = os.path.join(out_dir, 'quem_aumentou_eficiente_resumo.csv')
        pd.DataFrame(columns=['instalacao_nome','int_A','int_B','delta','n_A','n_B']).to_csv(eff_csv, index=False)
        return p_inc, p_red, eff_csv
    g['delta'] = g['int_B'] - g['int_A']
    eff_csv = os.path.join(out_dir, 'quem_aumentou_eficiente_resumo.csv')
    g.reset_index().to_csv(eff_csv, index=False)
    inc = g.sort_values('delta', ascending=False).head(k).copy()
    red = g.sort_values('delta', ascending=True).head(k).copy()
    p_inc = _plot_bar_delta(inc, 'Quem aumentou a intensidade (Δ>0) — 2020–2022 vs 2023–2025', 'fig_quem_aumentou.png', out_dir, color='#d62728', xmax=50.0, side='pos')
    p_red = _plot_bar_delta(red, 'Quem reduziu a intensidade (Δ<0) — 2020–2022 vs 2023–2025', 'fig_quem_reduziu.png', out_dir, color='#2ca02c', xmax=50.0, side='neg')
    return p_inc, p_red, eff_csv

# -------------------- 10) Validação 80/20 por instalação ------------------

def validate_80_20_by_installation(df, out_dir):
    """Validação simples 80/20 por instalação (baseline de média do treino)."""
    ensure_dir(out_dir)
    d = drop_both_zero(df).copy()
    if 'instalacao_nome' not in d.columns or 'Mes' not in d.columns:
        p_val = os.path.join(out_dir, 'validacao_80_20_placeholder.png')
        plt.figure(figsize=(8,4)); plt.text(0.5,0.5,'Colunas ausentes (instalacao_nome/Mes)',ha='center',va='center')
        plt.axis('off'); plt.savefig(p_val, dpi=200); plt.close()
        m_csv = os.path.join(out_dir, 'validacao_80_20_metricas.csv')
        pd.DataFrame(columns=['instalacao_nome','MAE','MAPE','N_teste']).to_csv(m_csv, index=False)
        return p_val, m_csv
    d = d[d['gas_produzido']>0].copy()
    d['intensidade_pct'] = (d['gas_queimado']/d['gas_produzido'])*100.0
    d = d[(d['intensidade_pct']>=0) & (d['intensidade_pct']<=100)].copy()
    d = d.sort_values(['instalacao_nome','Mes']).copy()
    rows = []
    for inst, sub in d.groupby('instalacao_nome'):
        n = len(sub)
        if n < 5:
            continue
        cut = int(np.ceil(n*0.8))
        train = sub.iloc[:cut]
        test = sub.iloc[cut:]
        if len(test)==0:
            continue
        y_train = train['intensidade_pct'].values
        y_test = test['intensidade_pct'].values
        mu = np.nanmean(y_train) if len(y_train)>0 else np.nan
        yhat = np.full_like(y_test, fill_value=mu, dtype=float)
        err = y_test - yhat
        mae = float(np.nanmean(np.abs(err)))
        denom = np.where(y_test==0, np.nan, y_test)
        mape = float(np.nanmean(np.abs(err)/denom)*100.0)
        rows.append({'instalacao_nome': inst, 'MAE': mae, 'MAPE': mape, 'N_teste': int(len(y_test))})
    met = pd.DataFrame(rows)
    m_csv = os.path.join(out_dir, 'validacao_80_20_metricas.csv')
    if met.empty:
        met = pd.DataFrame(columns=['instalacao_nome','MAE','MAPE','N_teste'])
    met.to_csv(m_csv, index=False)
    plt.figure(figsize=(10,5))
    if not met.empty:
        mae_vals = met['MAE'].astype(float).values
        med_mae = float(np.nanmedian(mae_vals))
        med_mape = float(np.nanmedian(met['MAPE'].astype(float).values)) if 'MAPE' in met.columns else float('nan')
        plt.hist(mae_vals, bins=20, color='#1f77b4', edgecolor='black', alpha=0.75)
        plt.xlabel('MAE (p.p. de intensidade)')
        plt.ylabel('Frequência')
        plt.title(f'Validação 80/20 por instalação — MAE mediana={med_mae:.2f} p.p.; MAPE mediana={med_mape:.1f}% (N_inst={len(met)})')
        plt.grid(True, ls=':', alpha=0.5)
    else:
        plt.text(0.5,0.5,'Sem instalações suficientes para validação 80/20',ha='center',va='center')
        plt.axis('off')
    p_val = os.path.join(out_dir, 'validacao_80_20_fig.png')
    plt.tight_layout(); plt.savefig(p_val, dpi=200); plt.close()
    return p_val, m_csv

# --------------------- Resolvedor de caminho de dados ---------------------

def _resolve_data_path(path_hint: str) -> str:
    """Resolve o caminho do dataset de forma amigável para repositórios públicos.

    Ordem de tentativa:
      1) `path_hint` (se existir);
      2) variável de ambiente `TCC_DATA_PATH` (se definida e existir);
      3) arquivo em `./data/` (padrão recomendado para GitHub);
      4) mesmo diretório do script e subpasta `data/`;
      5) qualquer arquivo `data_ANP*.xlsx|.csv` (ou legado `Dados_Finais*.xlsx|.csv`) nos diretórios acima.

    Dica (estrutura sugerida):
      repo/
        data/data_ANP.xlsx
        src/seu_script.py
    """
    # 1) Caminho fornecido diretamente
    if path_hint and os.path.exists(path_hint):
        return path_hint

    # 2) Variável de ambiente (opcional)
    env_path = os.environ.get("TCC_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Diretórios candidatos (cwd + pasta do script)
    candidates_dirs = []

    # cwd
    try:
        cwd = os.getcwd()
        candidates_dirs.extend([cwd, os.path.join(cwd, "data")])
    except Exception:
        pass

    # diretório do script (quando disponível)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates_dirs.extend([here, os.path.join(here, "data")])
    except Exception:
        here = None

    # 3/4) nomes preferenciais
    preferred_files = [
        os.path.join("data", "data_ANP.xlsx"),
        os.path.join("data", "data_ANP.csv"),
        "data_ANP.xlsx",
        "data_ANP.csv",
    ]
    for cand in preferred_files:
        if os.path.exists(cand):
            return cand

    # 5) busca por "data_ANP*" (xlsx/csv) nos diretórios candidatos (com suporte ao legado "Dados_Finais*")
    exts = (".xlsx", ".xls", ".csv")
    seen = set()
    for d in candidates_dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        try:
            for fn in os.listdir(d):
                fn_l = fn.lower()
                if fn_l.startswith(("data_anp", "dados_finais")) and fn_l.endswith(exts):
                    return os.path.join(d, fn)
        except Exception:
            continue

    raise FileNotFoundError(
        "Não encontrei o arquivo de dados. Para publicar no GitHub, recomenda-se colocar "
        "'data_ANP.xlsx' (ou .csv) em './data/'. Alternativamente, defina a variável "
        "de ambiente TCC_DATA_PATH com o caminho completo do arquivo."
    )

# -------------------------------- Orquestrador ----------------------------

def run_all(data_path=DATA_PATH, out_dir=OUT_DIR, sheet=SHEET_NAME):
    """Executa o pipeline completo do TCC e salva as figuras/arquivos em `out_dir`."""
    ensure_dir(out_dir)
    figs_dir = os.path.join(out_dir, "figs")
    ensure_dir(figs_dir)

    print("[INFO] Lendo planilha/CSV...")
    df = load_and_prepare(_resolve_data_path(data_path), sheet)

    print("[INFO] Série de intensidade (MM12)...")
    p_int = fig_intensidade_offshore(df, figs_dir)

    print("[INFO] Série sem meses com produção>0 e queima=0...")
    p_int_sem, n_sem = fig_intensidade_offshore_sem_queima(df, figs_dir)

    print("[INFO] Intensidade por tipo...")
    p_tipo = fig_intensidade_por_tipo(df, figs_dir)

    print("[INFO] Reinjecao x Intensidade (scatter)...")
    p_reinj, p_reinj_zoom, n_out, xlsx_path = fig_reinjecao_vs_intensidade(df, figs_dir)

    print("[INFO] Correlação reinjeção x intensidade...")
    p_cor, cor_csv = resumo_correlacao_reinj_intensidade(df, figs_dir)

    print("[INFO] Log–log GQ x GP...")
    p_log = fig_log_gq_vs_gp_simple(df, figs_dir)

    print("[INFO] Projeções 2030 (Y=0–10%)...")
    p_a, p_b, p_c = figs_projecoes_janelas(df, figs_dir, y_lim=(0, 10), clip_0_100=False)

    print("[INFO] Top 10 por Gás Queimado (absoluto)...")
    p_top10 = fig_top10_gq_por_campo(df, figs_dir)

    print("[INFO] Top 10 por Intensidade Média (normalizado)...")
    p_top10_norm = fig_top10_intensidade_media_por_instalacao(df, figs_dir, k=TOPK_INST)

    print("[INFO] Quem aumentou vs reduziu...")
    p_inc, p_red, eff_csv = figs_quem_aumentou_e_reduziu(df, figs_dir, k=TOPK_INST)

    print("[INFO] Validação 80/20 por instalação...")
    p_val, m_val = validate_80_20_by_installation(df, figs_dir)

    out = {
        "figs_dir": figs_dir,
        "figures": {
            "intensidade_offshore_serie_mm12": p_int,
            "intensidade_offshore_serie_mm12_sem_queima": p_int_sem,
            "intensidade_por_tipo": p_tipo,
            "reinjecao_vs_intensidade": p_reinj,
            "correlacao_reinjecao_vs_intensidade": p_cor,
            "reinjecao_vs_intensidade_zoom": p_reinj_zoom,
            "log_gq_vs_gp": p_log,
            "proj_2020_2023": p_a,
            "proj_2023_2025": p_b,
            "proj_2020_2025": p_c,
            "top10_gas_queimado": p_top10,
            "top10_intensidade_media": p_top10_norm,
            "quem_aumentou": p_inc,
            "quem_reduziu": p_red,
            "validacao_80_20_fig": p_val,
        },
        "arquivos": {
            "reinjecao_intensidade_maiores_que_100.xlsx": xlsx_path,
            "correlacao_reinj_intensidade_resumo.csv": cor_csv,
            "validacao_80_20_metricas.csv": m_val,
            "quem_aumentou_eficiente_resumo.csv": eff_csv,
            "paleta_campo_json": PALETA_JSON,
            "intensidade_offshore_sem_queima_excluidos.csv": os.path.join(
                figs_dir, "fig_intensidade_offshore_sem_queima_excluidos.csv"
            ),
            "intensidade_por_tipo_resumo.csv": os.path.join(
                figs_dir, "intensidade_por_tipo_resumo.csv"
            ),
        },
        "diagnosticos": {
            "n_outliers_>100pct": int(n_out),
            "n_meses_producao_sem_queima_excluidos": int(n_sem),
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out

if __name__ == '__main__':
    run_all()