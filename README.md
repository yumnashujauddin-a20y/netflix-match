# Netflix Match — Content-Based Recommendation System

✨ Overview
Netflix Match is an interactive movie and TV-show discovery
application. Select a title from the catalogue and explore
recommendations ranked by how closely their available metadata matches.

The project combines a Streamlit interface with a content-based
recommendation workflow to make catalogue exploration simple and
engaging.
Project note: Netflix Match is an independent educational project
and is not affiliated with Netflix. Similarity scores represent
metadata similarity; they are not audience ratings or guarantees of
personal preference.

🚀 Key Features
  Feature                             Description
  🔐 Login interface                  Login screen before accessing the
dashboard.

  🎯 Similar-title discovery          Find movies and shows related to a
selected title.

  🧠 Content-based matching           Uses TF-IDF feature vectors and
cosine similarity.

  🎛️ Recommendation controls          Adjust the number of results and
optionally restrict content type.

  🔎 Catalogue exploration            Search and filter titles in the
dataset.

  📥 Data export                      Download supported recommendation
or catalogue results as CSV.

  📊 Model evaluation                 View the evaluation information
provided by the project.
🎨 Custom interface                 Dark, cinematic Streamlit design
with optional login background
image.
🖥️ Application Preview
The application is organised around a simple user journey:
1. Open the application and sign in.
2. Select a movie or TV show.
3. Generate similar-title recommendations.
4. Explore the catalogue and evaluation information.
🗂️ Project Structure
Netflix Content Recommendation/
│
├── app.py                    # Streamlit application and user interface
├── Dataset.csv               # Movie and TV-show catalogue
├── recommender.py            # Recommendation logic
├── evaluate.py               # Evaluation script
├── evaluation_results.json   # Saved evaluation results, if generated
├── project_manifest.json     # Project metadata
├── PROJECT_REPORT.md         # Detailed project report
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
└── assets/
    └── netflix_bg.jpg         # Optional login background image
Keep app.py, recommender.py, and Dataset.csv in the same project
directory unless the paths in the application are changed.

🧰 Technology Stack
- Python --- application programming language
- Streamlit --- interactive web application interface
- pandas --- tabular data handling
- TF-IDF --- text feature extraction
- Cosine similarity --- similarity comparison
- CSV / JSON --- dataset and evaluation data formats
The exact installed packages and versions are listed in
requirements.txt.
⚠️ Limitations
- Recommendations are based on catalogue metadata, not individual
  viewing histories.
- Similarity scores describe metadata closeness, not predicted
  enjoyment.
- Results depend on the completeness and quality of the dataset.
- Evaluation results should be interpreted according to the
  methodology implemented in evaluate.py.
🔮 Possible Future Improvements
- User accounts with secure authentication
- Personal watchlists and favourites
- User feedback to improve recommendations
- Richer movie and show information
- Poster artwork and additional catalogue filters
- Hybrid recommendations combining content metadata and user
  preferences
📄 Project Files
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) --- project report
- [`requirements.txt`](requirements.txt) --- dependencies
- [`evaluate.py`](evaluate.py) --- evaluation implementation
- [`recommender.py`](recommender.py) --- recommendation logic
🙌 Acknowledgements
This project was created for learning and demonstrating content-based
recommendation techniques with Python and Streamlit.

Before publishing or redistributing the project, add the appropriate
dataset attribution and software licence."# netflix-match" 
