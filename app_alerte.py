import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv 
from pathlib import Path

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Bourse AI Analyst",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CHARGEMENT ROBUSTE DU .ENV (LOCAL) ---
try:
    script_dir = Path(__file__).parent
    env_path = script_dir / '.env'
    load_dotenv(dotenv_path=env_path)
except Exception:
    pass 

# --- CSS POUR LE LOOK MOBILE ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION DE NETTOYAGE DE CLÉ ---
def clean_key(key):
    if key:
        return key.strip().strip("'").strip('"')
    return None

# --- RÉCUPÉRATION DES SECRETS ---
raw_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        raw_key = st.secrets["GEMINI_API_KEY"]
except:
    pass

if not raw_key:
    raw_key = os.getenv("GEMINI_API_KEY")

# Interface de secours pour la clé
if not raw_key:
    with st.sidebar:
        st.warning("⚠️ Clé API non détectée")
        raw_key = st.text_input("Collez votre clé API Gemini ici :", type="password")

GEMINI_API_KEY = clean_key(raw_key)
UTILISER_IA = True

# --- CONFIGURATION IA ---
if UTILISER_IA and GEMINI_API_KEY:
    if not GEMINI_API_KEY.startswith("AIza"):
        model = None
        ia_status = "⚠️ Clé invalide"
        if GEMINI_API_KEY:
            st.error("🚨 La clé fournie ne semble pas valide (ne commence pas par 'AIza').")
    else:
        try:
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-pro')
            ia_status = "✅ IA Connectée"
        except Exception as e:
            model = None
            ia_status = f"❌ Erreur Config IA"
            if "API key expired" in str(e):
                st.error("🚨 VOTRE CLÉ API A EXPIRÉ.")
            else:
                st.error(f"Erreur IA : {e}")
else:
    model = None
    ia_status = "⚠️ IA en pause"

# --- FONCTIONS DU SYSTÈME ---
def check_cross(series_a, series_b, i):
    if i <= 0: return 0
    if series_a.iloc[i-1] < series_b.iloc[i-1] and series_a.iloc[i] > series_b.iloc[i]:
        return 1
    if series_a.iloc[i-1] > series_b.iloc[i-1] and series_a.iloc[i] < series_b.iloc[i]:
        return -1
    return 0

def check_slope(series, i):
    if i <= 0: return 0
    return 1 if series.iloc[i] > series.iloc[i-1] else -1

def calculate_indicators(df):
    for span in [5, 8, 10, 21]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    low_min = df['Low'].rolling(window=15).min()
    high_max = df['High'].rolling(window=15).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=5).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    aroon_window = 14
    df['Aroon_Up'] = df['High'].rolling(window=aroon_window).apply(lambda x: 100 * (np.argmax(x) + 1) / aroon_window)
    df['Aroon_Down'] = df['Low'].rolling(window=aroon_window).apply(lambda x: 100 * (np.argmin(x) + 1) / aroon_window)
    df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']
    return df

@st.cache_data(ttl=3600)
def analyze_market(tickers):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    
    for idx, ticker_symbol in enumerate(tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyse de {ticker_symbol}...")
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="6mo")
            if df.empty: continue
            
            df = calculate_indicators(df)
            i = len(df) - 1
            current_date = df.index[i].strftime('%Y-%m-%d')
            price = df['Close'].iloc[i]

            # Trigger EMA
            ema_trigger = 0
            ema_details = ""
            ema5 = df['EMA_5']
            for p in [8, 10, 21]:
                cross = check_cross(ema5, df[f'EMA_{p}'], i)
                if cross != 0:
                    ema_trigger = cross 
                    direction = "HAUSSIER" if cross == 1 else "BAISSIER"
                    ema_details = f"EMA 5 croise EMA {p} ({direction})"
                    break 
            
            if ema_trigger == 0: continue

            target_direction = ema_trigger
            direction_str = "ACHAT 🟢" if target_direction == 1 else "VENTE 🔴"

            # Filter Stoch
            stoch_k = df['Stoch_K'].iloc[i]
            stoch_d = df['Stoch_D'].iloc[i]
            stoch_ok = False
            if target_direction == 1 and stoch_k > stoch_d: stoch_ok = True
            elif target_direction == -1 and stoch_k < stoch_d: stoch_ok = True
            if not stoch_ok: continue

            # Confirmations
            confirmations = 0
            tech_details = [ema_details, "Stochastique validé"]

            if (target_direction == 1 and df['MACD_Line'].iloc[i] > df['MACD_Signal'].iloc[i]) or \
               (target_direction == -1 and df['MACD_Line'].iloc[i] < df['MACD_Signal'].iloc[i]):
                confirmations += 1
                tech_details.append("MACD")

            for ind_name, series in [('CCI', df['CCI']), ('Aroon', df['Aroon_Osc']), 
                                     ('CMF', df['CMF']), ('OBV', df['OBV']), ('RSI', df['RSI'])]:
                if check_slope(series, i) == target_direction:
                    confirmations += 1
                    tech_details.append(ind_name)

            if confirmations >= 5:
                results.append({
                    "ticker": ticker_symbol,
                    "price": price,
                    "date": current_date,
                    "signal": direction_str,
                    "tech_details": tech_details,
                    "confirmations": confirmations
                })
        except Exception:
            continue
            
    status_text.empty()
    progress_bar.empty()
    return results

def get_ai_advice(ticker, price, signal, details):
    if not model: return "IA non disponible"
    prompt = f"""
    Agis comme un analyste financier expert.
    ACTIF: {ticker} (Prix: {price})
    SIGNAL: {signal} (Stratégie EMA + Stochastique)
    DÉTAILS: {', '.join(details)}
    
    En 3 phrases maximum :
    1. Secteur de l'entreprise.
    2. Pourquoi ce signal technique est intéressant maintenant.
    3. Recommandation courte.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "API key expired" in str(e): return "⚠️ CLÉ EXPIRÉE."
        return f"Erreur IA : {str(e)}"

# --- INTERFACE UTILISATEUR (MOBILE) ---

st.title("📱 Trading Assistant")
st.caption(f"Scanner Pro | {ia_status}")

# Liste par défaut
DEFAULT_TICKERS = [
        "BMO.TO", "NA.TO", "RY.TO", "TD.TO", "CP.TO", "CCL-B.TO", "DOL.TO", "EMP-A.TO",
        "X.TO", "IAG.TO", "IFC.TO", "L.TO", "MRU.TO", "QBR-B.TO", "RBA.TO", "QSR.TO",
        "STN.TO", "TRI.TO", "TOI.V", "TIH.TO", "WCN.TO",
        "AMTM", "AXP", "AMP", "AMGN", "AAPL", "ANET", "AZN", "BX", "CMG", "COST",
        "CSX", "DKS", "DD", "FIS", "IBM", "J", "KIM", "KR", "LLY", "MRVL", "MA",
        "META", "MSFT", "NFLX", "NVDA", "ORLY", "ORCL", "PSX", "PG", "Q", "PWR",
        "DGX", "SHEL", "SO", "TMUS", "TSM", "TXN", "TOL", "TT", "SPY", "B",
        "ADBE", "GOOGL", "ADI", "AZO", "CARR", "CME", "CL", "CPRT", "FDS", "GGG",
        "HLT", "JNJ", "LIN", "LOW", "MTD", "MCO", "MSCI", "NKE", "OTIS", "PEP",
        "SHW", "TJX", "UNH", "ZTS", "VTI"
]

with st.expander("⚙️ Configuration & Liste", expanded=False):
    st.markdown("### 1. Liste de surveillance")
    selected_standard = st.multiselect(
        "Sélectionnez les actions favorites", 
        DEFAULT_TICKERS, 
        default=DEFAULT_TICKERS
    )
    
    st.markdown("### 2. Ajouter manuellement")
    st.markdown("Tapez d'autres symboles (séparés par une virgule).")
    st.caption("Ex: `TSLA, BTC-USD, SHOP.TO`")
    custom_input = st.text_input("Nouveaux Tickers", "")
    
    # Fusion des listes
    custom_list = []
    if custom_input:
        custom_list = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    
    final_tickers = list(set(selected_standard + custom_list))
    
    st.info(f"📊 Total : {len(final_tickers)} actifs à scanner")

    
if st.button("🚀 LANCER L'ANALYSE"):
    if not final_tickers:
        st.warning("Aucune action sélectionnée.")
    else:
        results = analyze_market(final_tickers)
        
        if not results:
            st.info("Aucun signal détecté sur la sélection.")
        else:
            st.success(f"{len(results)} opportunités trouvées !")
            
            for res in results:
                color = "green" if "ACHAT" in res['signal'] else "red"
                
                with st.container():
                    st.markdown(f"### {res['ticker']} : {res['signal']}")
                    col1, col2 = st.columns(2)
                    col1.metric("Prix", f"{res['price']:.2f}")
                    col2.metric("Confiance", f"{res['confirmations']}/6")
                    
                    with st.expander("🧠 Analyse IA & Détails"):
                        st.write("**Indicateurs validés :**")
                        st.code("\n".join(res['tech_details']))
                        
                        if model:
                            with st.spinner("L'IA réfléchit..."):
                                advice = get_ai_advice(res['ticker'], res['price'], res['signal'], res['tech_details'])
                                if "Erreur" in advice:
                                    st.error(advice)
                                else:
                                    st.info(advice)
                    st.divider()

st.markdown("---")
st.caption("Données Yahoo Finance. Trading risqué.")
                    
                    st.divider()

st.markdown("---")

st.caption("Données fournies par Yahoo Finance. Ceci n'est pas un conseil en investissement.")
