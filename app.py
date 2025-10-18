
import streamlit as st
import numpy as np, pandas as pd, pickle, json, io, tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓", layout="wide")


model  = tf.keras.models.load_model("heart.h5", compile=False)
scaler = pickle.load(open("scaler.pkl","rb"))

FALLBACK_FEATURES = [
    "Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak",
    "Sex_M",
    "ChestPainType_ATA","ChestPainType_NAP","ChestPainType_ASY",
    "RestingECG_Normal","RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat","ST_Slope_Up"
]
FEATURES = list(getattr(scaler, "feature_names_in_", [])) or FALLBACK_FEATURES

st.title("Heart Disease Predictor")
st.caption(f"model expects: {model.input_shape[-1]} | scaler expects: {getattr(scaler,'n_features_in_',None)}")
assert model.input_shape[-1] == 15
assert getattr(scaler,"n_features_in_",None) == 15


def build_dummy_row_from_raw(raw):
    """
    raw dict with original human-friendly fields:
    Age, RestingBP, Cholesterol, FastingBS(Yes/No), MaxHR, Oldpeak,
    Sex (Male/Female), ChestPainType (TA/ATA/NAP/ASY ...),
    RestingECG (LVH/Normal/ST-T wave abnormality),
    ExerciseAngina (Y/N), ST_Slope (Down/Flat/Up)
    returns a dict of 15 features in FEATURES order.
    """
    row = {c: 0.0 for c in FEATURES}

    
    def set_if(name, v):
        if name in row:
            row[name] = float(v)

    set_if("Age", raw["Age"])
    set_if("RestingBP", raw["RestingBP"])
    set_if("Cholesterol", raw["Cholesterol"])
    set_if("FastingBS", 1.0 if str(raw["FastingBS"]).strip().lower() in ("1","yes","y","true") else 0.0)
    set_if("MaxHR", raw["MaxHR"])
    set_if("Oldpeak", raw["Oldpeak"])

    
    if "Sex_M" in row:
        row["Sex_M"] = 1.0 if str(raw["Sex"]).strip().lower() in ("male","m","1") else 0.0

    
    cp_norm = str(raw["ChestPainType"]).strip().upper()
    cp_key = "ATA" if "ATA" in cp_norm else "NAP" if "NAP" in cp_norm else "ASY" if "ASY" in cp_norm else "TA"
    for c in ["ChestPainType_ATA","ChestPainType_NAP","ChestPainType_ASY"]:
        if c in row: row[c] = 1.0 if c.endswith(cp_key) else 0.0

    
    recg = str(raw["RestingECG"]).strip().upper()
    recg_key = "NORMAL" if "NORMAL" in recg else "ST" if "ST" in recg else "LVH"
    for c in ["RestingECG_Normal","RestingECG_ST"]:
        if c in row: row[c] = 1.0 if c.upper().endswith(recg_key) else 0.0

    
    if "ExerciseAngina_Y" in row:
        row["ExerciseAngina_Y"] = 1.0 if str(raw["ExerciseAngina"]).strip().upper() in ("Y","YES","1","TRUE") else 0.0

    
    slp = str(raw["ST_Slope"]).strip().upper()
    slp_key = "FLAT" if "FLAT" in slp else "UP" if "UP" in slp else "DOWN"
    for c in ["ST_Slope_Flat","ST_Slope_Up"]:
        if c in row: row[c] = 1.0 if c.upper().endswith(slp_key) else 0.0

    return row

def ensure_15_features_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
    
    
    cols = [c.lower() for c in df_in.columns]
    has_raw = all(x in cols for x in
                  ["age","restingbp","cholesterol","fastingbs","maxhr","oldpeak",
                   "sex","chestpaintype","restingecg","exerciseangina","st_slope"])

    if has_raw:
        rows = []
        for _, r in df_in.iterrows():
            raw = {
                "Age": r["Age"],
                "RestingBP": r["RestingBP"],
                "Cholesterol": r["Cholesterol"],
                "FastingBS": r["FastingBS"],
                "MaxHR": r["MaxHR"],
                "Oldpeak": r["Oldpeak"],
                "Sex": r["Sex"],
                "ChestPainType": r["ChestPainType"],
                "RestingECG": r["RestingECG"],
                "ExerciseAngina": r["ExerciseAngina"],
                "ST_Slope": r["ST_Slope"],
            }
            rows.append(build_dummy_row_from_raw(raw))
        X_df = pd.DataFrame(rows)
    else:
        
        X_df = df_in.copy()
        for c in FEATURES:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X_df = X_df[FEATURES]

    
    return X_df.astype(float)

def predict_proba_df(X_df: pd.DataFrame) -> np.ndarray:
    Xs = scaler.transform(X_df.values)
    p = model.predict(Xs, verbose=0).ravel()
    return p


tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📥 Bulk Predict (CSV)", "ℹ️ Model Information"])


with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age  = st.number_input("Age", 18, 110, 45)
        rbp  = st.number_input("RestingBP", 70, 240, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 80, 700, 200)
        fbs  = st.selectbox("FastingBS > 120 mg/dl", ["No","Yes"])
        mhr  = st.number_input("MaxHR", 60, 230, 150)
        op   = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    with col2:
        sex  = st.selectbox("Sex", ["Female","Male"])
        cp   = st.selectbox("Chest Pain Type", ["TA (Typical)","ATA (Atypical)","NAP (Non-anginal)","ASY (Asymptomatic)"])
        recg = st.selectbox("RestingECG", ["LVH","Normal","ST-T wave abnormality"])
        exg  = st.selectbox("ExerciseAngina", ["N","Y"])
        slope= st.selectbox("ST_Slope", ["Down","Flat","Up"])

    raw = {
        "Age": age, "RestingBP": rbp, "Cholesterol": chol, "FastingBS": fbs,
        "MaxHR": mhr, "Oldpeak": op, "Sex": sex,
        "ChestPainType": cp.split()[0],  # TA/ATA/NAP/ASY
        "RestingECG": recg,
        "ExerciseAngina": exg, "ST_Slope": slope
    }
    row = build_dummy_row_from_raw(raw)
    X_df = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)

    with st.expander("Debug: 15-feature row"):
        st.write(X_df)

    if st.button("Predict", type="primary"):
        p = float(predict_proba_df(X_df)[0])
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="th1")
        label = "✅ No Heart Disease" if p < threshold else "⚠️ High Risk of Heart Disease"
        (st.success if p < threshold else st.warning)(label)
        st.caption(f"Raw probability: {p:.3f} | Threshold: {threshold:.2f}")


with tab2:
    
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df_in = pd.read_csv(file)
        st.write("Input preview:", df_in.head())
        try:
            X_df = ensure_15_features_from_df(df_in)
            st.write("Model features preview (15 cols):", X_df.head())
            p = predict_proba_df(X_df)
            out = df_in.copy()
            out["probability"] = p
            out["prediction"]  = (p >= 0.5).astype(int)

            st.success(f"Predicted {len(out)} rows.")
            st.write(out.head())

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download predictions CSV", data=buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Processing error: {e}")


with tab3:
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Metrics")
        
        metrics = None
        try:
            metrics = json.load(open("metrics.json"))
        except Exception:
            pass

        if metrics:
            st.write(metrics)
        else:
            st.info("PERFORMANCE OF MODEL")

    with colB:
        st.subheader("Evaluate from labeled CSV")
        file2 = st.file_uploader("Upload labeled CSV", type=["csv"], key="eval_csv")
        if file2 is not None:
            df_eval = pd.read_csv(file2)
            if "HeartDisease" not in df_eval.columns:
                st.error("CSV: HeartDisease")
            else:
                y_true = df_eval["HeartDisease"].astype(int).values
                X_eval = ensure_15_features_from_df(df_eval.drop(columns=["HeartDisease"], errors="ignore"))
                p = predict_proba_df(X_eval)
                y_pred = (p >= 0.5).astype(int)

                acc = accuracy_score(y_true, y_pred)
                f1  = f1_score(y_true, y_pred)
                try:
                    auc = roc_auc_score(y_true, p)
                except:
                    auc = float("nan")

                st.write(f"Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  ROC-AUC: {auc:.3f}")
                st.text(classification_report(y_true, y_pred))

                
                cm = confusion_matrix(y_true, y_pred)
                fig = plt.figure()
                plt.imshow(cm, interpolation="nearest" , cmap ="Blues")
                plt.title("Confusion Matrix")
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ["No Disease", "Disease"])
                plt.yticks(tick_marks, ["No Disease", "Disease"])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, cm[i, j], ha="center", va="center")
                plt.xlabel("Predicted"); plt.ylabel("True")
                st.pyplot(fig)
