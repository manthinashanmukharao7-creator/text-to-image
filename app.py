import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch

st.set_page_config(
    page_title="CLIP Google Image Search",
    page_icon="🔍"
)

st.title("🔍 CLIP Text-Based Image Search")

st.write(
    "Search for images from the web and rank them using CLIP."
)

# -----------------------------
# Load CLIP model
# -----------------------------

@st.cache_resource
def load_model():

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return model, processor


model, processor = load_model()


# -----------------------------
# Google Image Search
# -----------------------------

def google_image_search(query, api_key, cx, num_images=10):

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "num": num_images
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error("Google Image Search API error.")
        return []

    data = response.json()

    images = []

    for item in data.get("items", []):

        image_url = item.get("link")

        if image_url:
            images.append(image_url)

    return images


# -----------------------------
# CLIP Ranking
# -----------------------------

def rank_images(query, image_urls):

    valid_images = []
    image_objects = []

    for url in image_urls:

        try:

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            image_objects.append(image)
            valid_images.append(url)

        except Exception:
            continue

    if not image_objects:
        return []

    # Text embedding
    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_features = model.get_text_features(
            **text_inputs
        )

    text_features = (
        text_features /
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    # Image embeddings
    image_inputs = processor(
        images=image_objects,
        return_tensors="pt"
    )

    with torch.no_grad():

        image_features = model.get_image_features(
            **image_inputs
        )

    image_features = (
        image_features /
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    # Similarity
    similarity = (
        text_features @ image_features.T
    )[0]

    scores, indices = torch.sort(
        similarity,
        descending=True
    )

    results = []

    for score, index in zip(scores, indices):

        results.append(
            (
                valid_images[index.item()],
                score.item()
            )
        )

    return results


# -----------------------------
# User interface
# -----------------------------

api_key = st.text_input(
    "Google API Key",
    type="password"
)

cx = st.text_input(
    "Google Custom Search Engine ID"
)

query = st.text_input(
    "Search images",
    placeholder="Example: golden retriever playing in snow"
)

num_images = st.slider(
    "Number of images",
    5,
    10,
    10
)


if st.button("🔍 Search"):

    if not api_key or not cx:

        st.warning(
            "Enter your Google API Key and Search Engine ID."
        )

    elif not query:

        st.warning(
            "Enter an image search query."
        )

    else:

        with st.spinner(
            "Searching Google Images..."
        ):

            image_urls = google_image_search(
                query,
                api_key,
                cx,
                num_images
            )

        if not image_urls:

            st.error(
                "No images were found."
            )

        else:

            with st.spinner(
                "Ranking images using CLIP..."
            ):

                results = rank_images(
                    query,
                    image_urls
                )

            st.subheader(
                "CLIP Ranked Results"
            )

            columns = st.columns(3)

            for i, (url, score) in enumerate(results):

                with columns[i % 3]:

                    st.image(
                        url,
                        use_container_width=True
                    )

                    st.write(
                        f"Similarity: {score:.4f}"
                    )