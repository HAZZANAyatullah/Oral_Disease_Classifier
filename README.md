# Oral_Disease_Detection
# 🦷 Dental Disease Classifier

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dental-disease-classifier.streamlit.app/)

A deep learning-powered web app built with **Streamlit** for detecting common dental diseases from oral images.  
The goal is to assist with **early detection**, raise awareness, and support healthcare workers in under-resourced communities.

---

## 🚀 Demo
👉 [Click here to try the app](https://dental-disease-classifier.streamlit.app/)

---

## 📌 Features
- Upload an oral image (tooth, gum, etc.).
- Classifies into possible dental conditions (e.g., **Caries, Gingivitis, Calculus**).
- Simple, lightweight interface for quick usage.
- Deployable via **Streamlit Cloud**.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** for the web app
- **PyTorch** for deep learning model
- **Torchvision / PIL** for image preprocessing
- **GitHub + Streamlit Cloud** for deployment

---

## 📂 Project Structure
├── app.py # Main Streamlit app

├── model/ # Trained model (not tracked in GitHub if >100MB)

├── requirements.txt # Dependencies

├── .gitignore # Ignore cache, venv, etc.

└── README.md # Project documentation


---

## ⚙️ Setup (Run Locally)
Clone the repo and install dependencies:

```bash
git clone https://github.com/HAZZANAyatullah/Oral_Disease_Detection.git
cd Oral_Disease_Detection
pip install -r requirements.txt
streamlit run app.py


📖 Future Improvements

Expand dataset for better accuracy.

Add more dental disease categories.

Multilingual support for wider accessibility.

Mobile-first interface.

👨‍⚕️ Author

Built with ❤️ by Ayatullah Hazzan
Passionate about leveraging **AI for oral healthcare innovation** and promoting **early disease detection** to improve community well-being

 



