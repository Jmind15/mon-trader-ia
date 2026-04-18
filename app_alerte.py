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
            # CORRECTION : gemini-3.1 n'est pas dispo, on utilise 1.5-flash qui est très rapide et stable
            model = genai.GenerativeModel('gemini-3.1-pro-preview')
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

# --- DICTIONNAIRE DES CATÉGORIES ---
TICKER_CATEGORIES = {
    "Main Watchlist": [
        "^DJI", "^IXIC", "^GSPC", "^VIX", "AAPL", "AMZN", "MSFT", "GOOG", "META", "TSLA", 
        "NVDA", "PLTR", "MSTR", "SPY", "QQQ", "TQQQ", "BRK-A", "BTC-USD", "DSG.TO"
    ],
    "Futures": [
        "GC=F", "^VIX", "YM=F", "NQ=F", "ES=F", "ALI=F", "BTC-USD", "HG=F", "CL=F", 
        "ETH-USD", "NG=F", "PA=F", "PL=F", "QQQ", "SI=F", "SPXU", "SPY", "TMF", "XLC", 
        "^DJI", "^NDX", "^NYA", "^RUT", "VIXY"
    ],
    "Dow Jones Components": [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", 
        "INTC", "JNJ", "KO", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", 
        "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW"
    ],
    "Dent USA": [
        "GWRE", "HEI-A", "ISRG", "MA", "MPWR", "PCOR", "SHW", "TDG", "TYL", "UBER", 
        "VEEV", "V", "VMC", "TMO", "MSI", "ECL", "DHR", "CSGP", "CDNS", "ASML", "MKL"
    ],
    "CAD": [
        "BMO.TO", "NA.TO", "RY.TO", "TD.TO", "CP.TO", "DOL.TO", "EMP-A.TO", "X.TO",
        "IAG.TO", "IFC.TO", "L.TO", "MRU.TO", "QBR-B.TO", "RBA.TO", "WCN.TO", "GIB-A.TO",
        "WSP.TO", "TOI.V", "VNP.TO", "CCL-B.TO", "PHYS.TO", "PSLV.TO", "SPPP.TO", "CIF.TO",
        "XEC.TO", "ZDB.TO", "PMM.TO", "EQL.TO", "NSCE.TO", "HTB.TO", "TRP.TO", "PPL.TO",
        "OXY", "EWC", "SHOP.TO"
    ],
    "Berkshire Hathaway": [
        "V", "KO", "AXP", "C", "SNOW", "AON", "OXY", "DHI", "COF", "LEN-B", "NU",
        "KR", "HPQ", "NVR", "MKL", "GL", "ALLY", "DVA", "JEF", "LPX"
    ],
    "COMMODITIES CAD": [
        "PHYS.TO", "PSLV.TO", "HURA.TO", "U.TO", "HUC.TO", "SPPP.TO", "OXY"
    ],
    "COMMODITIES USD": [
        "GLD", "SLV", "USO", "XLE", "SPPP", "PHYS", "KALU", "OXY", "COP"
    ],
    "Cryptos": [
        "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "XLM-USD", "SOL-USD", "COIN", "MSTR", "BMNR"
    ],
    "Pool": [
        "HUT.TO", "PTM.TO", "HUC.TO", "VDE", "CHOW", "CLS.TO", "ARE.TO", "WSPOF", "STN.TO", 
        "BDT.TO", "BIP", "BIP-UN.TO", "BAM", "TRP.TO", "AMT", "GEHC", "SHL.DE", "MDT", 
        "COIN", "MSTR", "EWY", "OVH.PA", "OVHFF", "BMNR"
    ],
    "CHF": [
        "NESN.SW", "BCVN.SW"
    ],
    "USD Growth": [
        "AMTM", "AXP", "AMP", "AMGN", "AAPL", "ANET", "AZN", "BX", "CMG", "COST",
        "CSX", "DKS", "DD", "FIS", "IBM", "MRVL", "MA", "META", "MSFT", "NFLX",
        "NVDA", "ORLY", "ORCL", "PSX", "PG", "Q", "PWR", "DGX", "SHEL", "SO",
        "TMUS", "TT", "SPY", "B", "TSLA", "GOOG", "GLD", "SLV", "SPPP", "GEHC",
        "XAR", "DFEN", "HOOD", "MCO", "BP", "TTE", "DJT", "CME", "AZO", "GGG", "ASML"
    ],
    "USD Value": [
        "ADBE", "GOOGL", "ADI", "AZO", "CARR", "CME", "CL", "CPRT", "FDS", "GGG",
        "HLT", "JNJ", "LIN", "LOW", "MRVL", "MCO", "MSCI", "NFLX", "NKE", "NVDA",
        "ORLY", "ORCL", "OTIS", "PEP", "PSX", "PG", "PWR", "DGX", "SHEL", "SHW",
        "SO", "TSM", "TXN", "TJX", "TOL", "TT", "UNH", "ZTS", "VTI"
    ]
}

with st.expander("⚙️ Configuration & Liste", expanded=False):
    st.markdown("### 1. Sélection par Catégorie")
    
    # L'utilisateur choisit les listes qu'il veut scanner (Main Watchlist par défaut)
    selected_categories = st.multiselect(
        "Listes à analyser", 
        options=list(TICKER_CATEGORIES.keys()), 
        default=["Main Watchlist"]
    )
    
    st.markdown("### 2. Ajouter manuellement")
    st.markdown("Tapez d'autres symboles (séparés par une virgule).")
    st.caption("Ex: `TSLA, BTC-USD, SHOP.TO`")
    custom_input = st.text_input("Nouveaux Tickers", "")
    
    # Préparation des ajouts manuels
    custom_list = []
    if custom_input:
        custom_list = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        
    total_tickers = sum([len(TICKER_CATEGORIES[cat]) for cat in selected_categories]) + len(custom_list)
    st.info(f"📊 Total : {total_tickers} actifs répartis à scanner")

    
if st.button("🚀 LANCER L'ANALYSE"):
    if not selected_categories and not custom_list:
        st.warning("Aucune action ou catégorie sélectionnée.")
    else:
        # Création d'un dictionnaire des catégories à analyser
        categories_to_process = {}
        for cat in selected_categories:
            categories_to_process[cat] = TICKER_CATEGORIES[cat]
        if custom_list:
            categories_to_process["Ajouts Manuels"] = custom_list
        
        total_opportunites = 0
        
        # On parcourt chaque catégorie séparément
        for cat_name, cat_tickers in categories_to_process.items():
            st.markdown(f"## 📂 Catégorie : {cat_name}")
            
            # Déduplication au sein de la catégorie
            cat_tickers = list(set(cat_tickers))
            
            results = analyze_market(cat_tickers)
            
            if not results:
                st.info(f"Aucun signal détecté dans la catégorie {cat_name}.")
            else:
                st.success(f"🎯 {len(results)} opportunité(s) trouvée(s) !")
                total_opportunites += len(results)
                
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
            
            st.markdown("<br>", unsafe_allow_html=True) # Espace visuel entre les catégories
        
        if total_opportunites > 0:
            st.balloons()
            st.markdown(f"### 🎉 Analyse terminée : {total_opportunites} signaux détectés au total !")

st.markdown("---")
st.caption("Données Yahoo Finance. Trading risqué.")
                    


