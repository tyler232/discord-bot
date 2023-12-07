import wikipediaapi
import os
import nltk
import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv

# Install internal package in nltk if not already
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except (LookupError, FileNotFoundError):
    print("NLTK data not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Load environment variables from .env file
load_dotenv()
user_agent = os.environ.get("WIKIPEDIA_USER_AGENT")


def wikipedia_summary(query):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')

    page_py = wiki_wiki.page(query)

    if page_py.exists():
        # Check if the page is likely a disambiguation page
        summary_lower = page_py.summary.lower()
        if any(keyword in summary_lower for keyword in ["refer", "may refer", "can refer"]):
            return f"The term '{query}' may refer to multiple topics. Please be more specific."
        else:
            first_paragraph = page_py.summary.split('\n\n')[0]
            word_count = len(nltk.word_tokenize(first_paragraph))
            print("Word Count:", word_count)

            # If the word count is larger than or equal to 500, truncate the summary
            if word_count >= 300:
                print("Reached Here 31")
                truncated_sum = ' '.join(first_paragraph.split()[:300])
                # Limit the content length to 500 characters
                truncated_sum = truncated_sum[:500]

                # Remove accents using unidecode
                truncated_sum = unidecode.unidecode(truncated_sum)

                # Remove any incomplete sentence at the end
                sentences = nltk.tokenize.sent_tokenize(truncated_sum)
                if len(sentences) > 1:
                    truncated_sum = ' '.join(sentences[:-1])

                return truncated_sum
            else:
                print("Reached Here 32")
                return first_paragraph
    else:
        return "I couldn't find information on that topic."

def find_most_viewed_page(options):
    # For simplicity, you can choose the most viewed page based on the number of links
    most_viewed_page_title = max(options, key=lambda option: len(wikipediaapi.Wikipedia(user_agent, 'en').page(option).links))
    most_viewed_page = wikipediaapi.Wikipedia(user_agent, 'en').page(most_viewed_page_title)
    return most_viewed_page

def extract_topic(message):
    # Tokenize the message and remove punctuation
    tokens = word_tokenize(message)
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Extract the topic
    topic = " ".join(tokens)
    return topic

