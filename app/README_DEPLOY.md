# Putting the demo online (optional bonus)

The Streamlit app is a small bonus: +1 for a working GUI and +1 for a public link. You can skip it without affecting the main marks.

## Run it on your own machine

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Run `notebooks/classification.ipynb` first so that `models/best_classifier.joblib` exists, or commit the saved model to the repo. The app has sliders and dropdowns for a person's details and shows the predicted income bracket with the probabilities.

## Make it public with Streamlit Community Cloud (free)

1. Push the repo to GitHub. It needs to be public, with `requirements.txt` at the root and `models/best_classifier.joblib` committed.
2. Go to https://share.streamlit.io and click **New app**, then pick your repo and branch.
3. Set the main file path to `app/app.py` and click **Deploy**.
4. Copy the URL you get into the "Live demo" line of the README, and have it open during the viva.

Hugging Face Spaces or Render also work if you prefer them.
