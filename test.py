import nltk
from nltk.corpus import stopwords

try:
    stop_words = stopwords.words('english')
    print("Stopwords are already downloaded ✅")
except LookupError:
    print("Stopwords not found ❌ — downloading now...")
    nltk.download('stopwords')