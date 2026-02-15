from flask import Flask, request, render_template
import joblib       
import pickle        
import os           


app = Flask(__name__)

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "static", "model")

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")           # Saved TF-IDF vectorizer
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")           # Saved ML model
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")   # Saved LabelEncoder

# -------------------- LOAD TRAINED OBJECTS --------------------
# Load TF-IDF vectorizer 
tfidf = joblib.load(TFIDF_PATH)

# Load LabelEncoder
label_encoder = joblib.load(LE_PATH)

# Load trained ML model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# -------------------- HOME PAGE ROUTE --------------------
@app.route("/", methods=["GET"])
def home():
    # Render index.html with empty prediction initially
    return render_template("index.html", prediction=None, text="")


# -------------------- WEB FORM PREDICTION --------------------
@app.route("/predict", methods=["POST"])
def predict():
    txt = request.form.get("text", "").strip()

    if not txt:
        return render_template(
            "index.html",
            prediction="Please enter text.",
            text=txt,
            nnz=None
        )

    # Convert text → numerical vector using trained TF-IDF
    x_vec = tfidf.transform([txt])

    # Predict category 
    pred = model.predict(x_vec)

    # Convert encoded number → actual category name
    pred_label = label_encoder.inverse_transform(pred)[0]

    # Send prediction result back to HTML page
    return render_template(
        "index.html",
        prediction=pred_label,
        text=txt,
        nnz=int(x_vec.nnz)  
    )


if __name__ == "__main__":
    # Run Flask app
    # debug=True is good for development (auto reload + error details)
    app.run(host="0.0.0.0", port=5000, debug=True)
