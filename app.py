from pathlib import Path

import pandas as pd
import streamlit as st
import hmac
import base64

from recommender import NetflixRecommender, evaluate_recommender, load_dataset

st.set_page_config(
    page_title="Netflix Match | Content Recommender",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# NETFLIX MATCH LOGIN
# This runs before loading the recommendation model.
# Demo credentials: admin / admin123
# For deployment, configure .streamlit/secrets.toml.
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def verify_login(username: str, password: str) -> bool:
    try:
        expected_username = st.secrets["auth"]["username"]
        expected_password = st.secrets["auth"]["password"]
    except Exception:
        expected_username = "admin"
        expected_password = "admin123"

    return (
        hmac.compare_digest(username, str(expected_username))
        and hmac.compare_digest(password, str(expected_password))
    )


if not st.session_state.logged_in:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {
            background:
              radial-gradient(circle at 18% 42%, rgba(91, 26, 122, .30), transparent 38%),
              radial-gradient(circle at 85% 18%, rgba(35, 35, 119, .20), transparent 34%),
              linear-gradient(120deg, #080719 0%, #10091f 55%, #080719 100%);
            color: #fff;
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }
        .block-container {
            max-width: 1450px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }
        .login-brand {
            font-size: clamp(34px, 4vw, 55px);
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -1.7px;
            color: #fff;
            margin: 8px 0 16px;
        }
        .login-brand .n-mark {
            color: #f20b35;
            font-size: 1.25em;
            text-shadow: 0 5px 24px rgba(242, 11, 53, .25);
        }
        .login-description {
            max-width: 550px;
            color: #d0cde0;
            font-size: 18px;
            line-height: 1.8;
            margin-bottom: 42px;
        }
        .login-feature {
            display: flex;
            align-items: center;
            gap: 18px;
            max-width: 530px;
            margin: 29px 0;
            padding: 12px 14px;
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 16px;
            background: rgba(255,255,255,.035);
        }
        .login-feature-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 62px;
            height: 62px;
            min-width: 62px;
            border-radius: 50%;
            font-size: 26px;
            background: linear-gradient(135deg, #9d20c4, #4339e8);
        }
        .login-feature-title {
            color: #fff;
            font-size: 16px;
            font-weight: 700;
        }
        .login-feature-sub {
            color: #aaa8c0;
            font-size: 13px;
            margin-top: 6px;
        }
        .login-card {
            padding: 30px 30px 22px;
            border: 1px solid rgba(153,139,255,.25);
            border-radius: 23px;
            background: rgba(17,17,43,.88);
            box-shadow: 0 20px 65px rgba(0,0,0,.42);
        }
        .login-card-title {
            text-align: center;
            color: #fff;
            font-size: 29px;
            font-weight: 800;
        }
        .login-card-title span { color: #f20b35; }
        .login-card-sub {
            text-align: center;
            color: #bcb8d0;
            font-size: 14px;
            margin: 8px 0 22px;
        }
        div[data-testid="stTextInput"] label {
            color: #e4e0f5 !important;
        }
        div[data-testid="stTextInput"] input {
            background: #171832 !important;
            color: white !important;
            border: 1px solid #383963 !important;
            border-radius: 11px !important;
            min-height: 47px;
        }
        div[data-testid="stCheckbox"] label { color: #ddd9ee !important; }
        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 49px;
            border: 0;
            border-radius: 12px;
            color: white;
            font-weight: 700;
            background: linear-gradient(90deg, #f20b35, #ad08b8);
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            color: white;
            border: 0;
            filter: brightness(1.12);
        }
        .login-footer {
            color: #aaa8c0;
            text-align: center;
            font-size: 13px;
            margin-top: 22px;
            line-height: 1.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # BACKGROUND IMAGE
    bg_path = Path(__file__).parent / "assets" / "zgeTuV.jpg"

    if bg_path.exists():
      bg_base64 = base64.b64encode(
        bg_path.read_bytes()
    ).decode("utf-8")

    st.markdown(f"""
    <style>
    .stApp {{
        background:
            linear-gradient(
                rgba(5,5,20,0.72),
                rgba(5,5,20,0.82)
            ),
            url("data:image/jpeg;base64,{bg_base64}");

        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown(
            """
            <div style="padding:30px 12px">
              <div class="login-brand"><span class="n-mark">N</span> Netflix Match</div>
              <div class="login-description">
                Discover your next watch through content similarity,
                genres, and metadata.
              </div>

              <div class="login-feature">
                <div class="login-feature-icon">🎬</div>
                <div>
                  <div class="login-feature-title">Personalized Recommendations</div>
                  <div class="login-feature-sub">Find titles that match your unique taste.</div>
                </div>
              </div>

              <div class="login-feature">
                <div class="login-feature-icon">🔎</div>
                <div>
                  <div class="login-feature-title">Explore by Similarity</div>
                  <div class="login-feature-sub">Discover content similar to what you love.</div>
                </div>
              </div>

              <div class="login-feature">
                <div class="login-feature-icon">📊</div>
                <div>
                  <div class="login-feature-title">Smart Insights</div>
                  <div class="login-feature-sub">Powered by genres, metadata, and AI.</div>
                </div>
              </div>

              <div style="color:#eee;font-size:16px;margin-top:40px">
                🎁 &nbsp; More of what you love. Less time searching.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-card">
              <div class="login-card-title"><span>N</span> Netflix Match</div>
              <div class="login-card-sub">Sign in to continue your journey</div>
              <div style="width:100px;height:5px;border-radius:10px;margin:0 auto 22px;
                   background:linear-gradient(90deg,#f20b35,#8d28d8)"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("netflix_match_login"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
            )
            password = st.text_input(
                "Password",
                placeholder="Enter your password",
                type="password",
            )
            st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Login  →", use_container_width=True)

            if submitted:
                if verify_login(username.strip(), password):
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")

        st.markdown(
            """
            <div class="login-footer">
              Sign in to explore movies and TV shows picked for you.<br>
              Netflix Match · Content-based recommendation demo
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Local demo credentials: admin / admin123")

    # Do not load the dataset or show the dashboard until login succeeds.
    st.stop()

# =========================================================
# END LOGIN — ORIGINAL RECOMMENDATION APP CONTINUES BELOW
# =========================================================


DATA_PATH = Path("Dataset.csv")

st.markdown(
    """
    <style>
    .main { background: #0b0b0f; }
    .block-container { padding-top: 1.6rem; max-width: 1250px; }
    .hero {
      padding: 1.7rem 1.9rem; border-radius: 18px;
      background: linear-gradient(120deg, #211018 0%, #14141c 55%, #241015 100%);
      border: 1px solid #38252d; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p { color: #c9c5ca; margin-bottom: 0; }
    .small-muted { color: #aaa6ad; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Preparing title features and similarity model…")
def get_recommender(path: str):
    return NetflixRecommender(load_dataset(path))

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}. Put Dataset.csv in the data folder.")
    st.stop()

try:
    model = get_recommender(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

df = model.df

st.markdown(
    """
    <div class="hero">
      <h1>🎬 Netflix Match</h1>
      <p>Discover your next watch through content similarity, genres, and metadata.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_titles = len(df)
movie_count = int(df["type"].astype(str).str.casefold().eq("movie").sum())
show_count = int(df["type"].astype(str).str.casefold().eq("tv show").sum())
genre_count = len({g.strip() for values in df["listed_in"].fillna("") for g in str(values).split(",") if g.strip()})

m1, m2, m3, m4 = st.columns(4)
m1.metric("Titles in catalogue", f"{total_titles:,}")
m2.metric("Movies", f"{movie_count:,}")
m3.metric("TV shows", f"{show_count:,}")
m4.metric("Unique categories", f"{genre_count:,}")

with st.sidebar:
    st.header("Recommendation settings")
    st.caption(f"Signed in as: {st.session_state.get('username', 'user')}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    st.divider()
    st.caption("Choose a title and tune the results.")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    restrict_type = st.toggle("Only recommend the same content type", value=True)
    st.divider()
    st.subheader("About this model")
    st.write("TF-IDF converts title metadata into vectors. Cosine similarity ranks titles with similar feature patterns.")
    st.caption("Scores compare metadata similarity; they are not ratings or probabilities of enjoyment.")

tab_recommend, tab_explore, tab_evaluate, tab_about = st.tabs(
    ["✨ Find similar titles", "🔎 Explore catalogue", "📊 Model evaluation", "ℹ️ Project details"]
)

with tab_recommend:
    left, right = st.columns([1.05, 1.3], gap="large")
    with left:
        title_choices = model.title_options()
        display_labels = title_choices.apply(lambda r: f"{r['title']} · {r['type']}", axis=1).tolist()
        selected_label = st.selectbox("Select a Netflix title", display_labels, index=0)
        selected_pos = display_labels.index(selected_label)
        selected_row = title_choices.iloc[selected_pos]
        selected_title = str(selected_row["title"])
        selected_type = str(selected_row["type"])

        matches = df[
            df["title"].astype(str).str.casefold().eq(selected_title.casefold())
            & df["type"].astype(str).str.casefold().eq(selected_type.casefold())
        ]
        source = matches.iloc[0] if not matches.empty else df[df["title"] == selected_title].iloc[0]

        st.markdown("#### Selected title")
        st.markdown(f"### {source['title']}")
        st.caption(f"{source.get('type', 'Unknown type')} · {source.get('release_year', 'Year unknown')}")
        st.write(f"**Categories:** {source.get('listed_in') or 'Not listed'}")
        st.write(f"**Director:** {source.get('director') or 'Not listed'}")
        st.write(f"**Country:** {source.get('country') or 'Not listed'}")
        st.write(f"**Rating:** {source.get('rating') or 'Not listed'} · **Duration:** {source.get('duration') or 'Not listed'}")
        run = st.button("Find similar titles", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Your recommendations")
        if run or "last_recommendations" not in st.session_state:
            try:
                recs = model.recommend(
                    selected_title,
                    top_n=top_n,
                    content_type=selected_type if restrict_type else None,
                )
                # If there are duplicate titles, the type selector resolves the source.
                st.session_state["last_recommendations"] = recs
                st.session_state["last_source"] = selected_title
            except Exception as exc:
                st.error(str(exc))
                recs = pd.DataFrame()
        else:
            recs = st.session_state["last_recommendations"]

        if not recs.empty:
            for rank, (_, rec) in enumerate(recs.iterrows(), start=1):
                with st.container(border=True):
                    c1, c2 = st.columns([0.12, 0.88])
                    c1.markdown(f"## {rank}")
                    c2.markdown(f"**{rec.get('title', 'Unknown title')}**")
                    c2.caption(
                        f"{rec.get('type', '')} · {rec.get('release_year', '')} · "
                        f"{rec.get('rating', 'Rating unknown')}"
                    )
                    c2.progress(min(max(float(rec.get("similarity", 0)), 0.0), 1.0))
                    c2.write(f"**Metadata similarity:** {rec.get('similarity_percent', 0)}%")
                    c2.write(f"**Categories:** {rec.get('listed_in') or 'Not listed'}")
                    c2.write(f"**Why it matched:** {rec.get('shared_categories', 'Shared metadata')}")
            export_cols = [c for c in ["title", "type", "release_year", "listed_in", "similarity_percent", "shared_categories"] if c in recs]
            st.download_button(
                "Download recommendations as CSV",
                recs[export_cols].to_csv(index=False).encode("utf-8"),
                file_name="netflix_recommendations.csv",
                mime="text/csv",
            )
        else:
            st.info("Choose a title and click **Find similar titles** to generate recommendations.")

with tab_explore:
    st.subheader("Explore the dataset")
    search = st.text_input("Search title names", placeholder="Try: Stranger Things")
    col1, col2, col3 = st.columns(3)
    types = sorted([x for x in df["type"].dropna().astype(str).unique() if x])
    type_filter = col1.selectbox("Content type", ["All"] + types)
    years = pd.to_numeric(df["release_year"], errors="coerce")
    min_year = int(years.min()) if years.notna().any() else 1900
    max_year = int(years.max()) if years.notna().any() else 2025
    year_range = col2.slider("Release year range", min_year, max_year, (min_year, max_year))
    genre_options = sorted({g.strip() for v in df["listed_in"].fillna("") for g in str(v).split(",") if g.strip()})
    genre_filter = col3.selectbox("Category", ["All"] + genre_options)

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]
    if type_filter != "All":
        filtered = filtered[filtered["type"] == type_filter]
    filtered_years = pd.to_numeric(filtered["release_year"], errors="coerce")
    filtered = filtered[filtered_years.between(year_range[0], year_range[1]) | filtered_years.isna()]
    if genre_filter != "All":
        filtered = filtered[filtered["listed_in"].str.contains(genre_filter, case=False, na=False)]

    display_cols = [c for c in ["title", "type", "release_year", "listed_in", "director", "country", "rating", "duration"] if c in filtered]
    st.caption(f"{len(filtered):,} matching titles")
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True, height=480)
    st.download_button(
        "Download filtered catalogue",
        filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_netflix_catalogue.csv",
        mime="text/csv",
    )

with tab_evaluate:
    st.subheader("Offline evaluation")
    st.write(
        "Because the dataset has no user ratings, watch history, or human relevance labels, "
        "this project reports a **genre-overlap proxy**. Treat it as a diagnostic, not proof of user satisfaction."
    )
    sample_size = st.slider("Titles to sample", 25, min(500, len(df)), min(100, len(df)), step=25)
    k = st.selectbox("Top-K cutoff", [3, 5, 10], index=1)
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating recommendations on a random sample…"):
            metrics = evaluate_recommender(model, sample_size=sample_size, k=k)
        a, b, c = st.columns(3)
        a.metric("Titles evaluated", metrics["titles_evaluated"])
        b.metric(f"Precision@{k} proxy", f"{metrics['precision_at_k_proxy']:.1%}")
        c.metric("Mean genre overlap", f"{metrics['mean_genre_overlap']:.1%}")
        st.info(metrics["note"])
        st.json(metrics)

with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Prepare metadata:** clean missing values and duplicate title/type pairs.
        2. **Feature engineering:** combine categories, type, director, country, rating, year, and duration.
        3. **Vectorization:** TF-IDF converts text metadata into numerical vectors.
        4. **Similarity:** cosine similarity ranks candidates by feature similarity.
        5. **Evaluation:** compare genre overlap on a reproducible sample.
        """
    )
    st.subheader("Important limitations")
    st.markdown(
        """
        - The uploaded dataset does not include plot descriptions, cast, user profiles, ratings, or watch history.
        - The recommender is content-based, so it does not learn an individual user's taste.
        - Similarity scores are relative metadata comparisons, not probabilities.
        - The genre-overlap evaluation is a proxy and should be supplemented with human review or interaction data.
        """
    )
    st.subheader("Dataset columns")
    st.dataframe(pd.DataFrame({"Column": df.columns, "Non-null values": [int(df[c].notna().sum()) for c in df.columns]}), hide_index=True, use_container_width=True)

st.divider()
st.markdown(
    '<p class="small-muted">Educational project · Built with Python, pandas, scikit-learn, and Streamlit. '
    'Netflix branding is used descriptively; this is an independent demo.</p>',
    unsafe_allow_html=True,
)
