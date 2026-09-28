import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="CLIP Text-Based Image Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 CLIP Text-Based Image Search")

st.write(
    "Search for images from the web and rank them using CLIP."
)


# --------------------------------
# Load CLIP
# --------------------------------

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


# --------------------------------
# Google Image Search
# --------------------------------

def google_image_search(query, number_of_images):

    api_key = st.secrets["GOOGLE_API_KEY"]
    search_engine_id = st.secrets["GOOGLE_CX"]

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "searchType": "image",
        "num": number_of_images
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    if response.status_code != 200:
        st.error("Google Image Search failed.")
        return []

    data = response.json()

    image_urls = []

    for item in data.get("items", []):

        image_url = item.get("link")

        if image_url:
            image_urls.append(image_url)

    return image_urls


# --------------------------------
# Download images
# --------------------------------

def download_images(image_urls):

    images = []
    valid_urls = []

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

            images.append(image)
            valid_urls.append(url)

        except Exception:
            continue

    return images, valid_urls


# --------------------------------
# CLIP ranking
# --------------------------------

def rank_images(query, images, urls):

    if len(images) == 0:
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

    # Image embedding
    image_inputs = processor(
        images=images,
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

        results.append({
            "url": urls[index.item()],
            "score": score.item()
        })

    return results


# --------------------------------
# User interface
# --------------------------------

query = st.text_input(
    "Search images",
    placeholder="Example: golden retriever playing in snow"
)

number_of_images = st.slider(
    "Number of images",
    min_value=5,
    max_value=10,
    value=10
)


if st.button("🔍 Search"):

    if query.strip() == "":

        st.warning(
            "Please enter an image search query."
        )

    else:

        # Search Google
        with st.spinner(
            "Searching images from the web..."
        ):

            image_urls = google_image_search(
                query,
                number_of_images
            )

        if len(image_urls) == 0:

            st.error(
                "No images found."
            )

        else:

            # Download
            with st.spinner(
                "Downloading images..."
            ):

                images, valid_urls = download_images(
                    image_urls
                )

            # CLIP ranking
            with st.spinner(
                "Ranking images using CLIP..."
            ):

                results = rank_images(
                    query,
                    images,
                    valid_urls
                )

            st.subheader(
                "🔍 CLIP Ranked Results"
            )

            columns = st.columns(3)

            for i, result in enumerate(results):

                with columns[i % 3]:

                    st.image(
                        result["url"],
                        use_container_width=True
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['score']:.4f}"
                    )
