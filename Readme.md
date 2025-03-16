# 🌿 **Skin Disease Detection AI Model (Flask + TensorFlow)**  

An AI-based **Skin Disease Prediction System** using **Deep Learning (CNN Model)** with **Flask API** and **Ngrok for Public URL**.

---

## 🛠️ **Tech Stack**
- Flask (Backend API)
- TensorFlow (Deep Learning Model)
- Keras
- Pyngrok (For Public URL)
- Bootstrap (Frontend UI)

---

## 📂 **Project Structure**
symptom_checker/
│
├── data/
│   └── images/
├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
└── models/
    └── skin_disease_model.h5




---

## ✅ **Step 1: Clone the Repository**
```bash
git clone https://github.com/omkar7075/skin-disease-detection.git
python3 -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python app.py

📤 Upload Image and Predict
Upload an image of a skin disease.
The model will predict the disease with Confidence Score and Treatment Details.
📄 Sample Output
Disease	Confidence	Causes	Treatment
Melanoma	96.5%	UV Exposure, Genetic Mutation	Surgical Removal
Basal Cell Carcinoma	89.7%	Chronic Sun Exposure	Mohs Surgery
🔍 Troubleshooting
Error	Solution
No file uploaded	Ensure the image file is uploaded
Model Not Loading	Check the model path in app.py
Ngrok URL not generated	Restart Ngrok service
🎯 Deployment on GitHub
bash
Copy
Edit
git add .
git commit -m "Initial Commit"
git push origin main
🚀 Demo URL
Live Demo Link

🛑 Stop Flask Server
bash
Copy
Edit
ctrl + c
📚 References
TensorFlow Documentation
Flask Official Docs
Ngrok Documentation
⭐️ Contribute
Feel free to contribute and improve the project.

📞 Contact Me
If you face any issues, DM me on LinkedIn or GitHub. 😊

yaml
Copy
Edit

---

### ✅ Now save this as `README.md` file in your project.

Would you like me to generate the final folder structure for your project? 😊







