import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def fetch_online_data(query: str, num_results: int = 5) -> List[str]:
    """
    Fetch online data using a search query.
    :param query: Search query.
    :param num_results: Number of results to fetch.
    """
    api_key = "YOUR_GOOGLE_API_KEY"  # Replace with your API key
    cse_id = "YOUR_CSE_ID"  # Replace with your CSE ID
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&num={num_results}"
    try:
        response = requests.get(url)
        results = response.json().get('items', [])
        formatted_results = [f"{item['title']}: {item['snippet']}" for item in results]
        logger.info(f"Online data fetched for query '{query}': {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"Error fetching online data: {e}")
        return []
