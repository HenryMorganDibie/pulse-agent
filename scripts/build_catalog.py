"""Build the full production catalog: real Goodreads books + multi-category items + Nigerian items."""
import json
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")

# Load existing Goodreads catalog
with open(CATALOG_PATH, encoding="utf-8") as f:
    catalog = json.load(f)

print(f"Existing catalog: {len(catalog)} items (all Books)")

# --- Multi-category items (Food, Nightlife, Movies, Shopping, Outdoors, Arts) ---
multi_cat = [
    # Food
    {"item_id": "i_001", "name": "The Grill House",          "category": "Food",      "attributes": {"cuisine": "American",  "price_range": "$$",  "vibe": "casual"}},
    {"item_id": "i_002", "name": "Sakura Ramen",             "category": "Food",      "attributes": {"cuisine": "Japanese",  "price_range": "$",   "vibe": "cozy"}},
    {"item_id": "i_003", "name": "La Bella Italia",          "category": "Food",      "attributes": {"cuisine": "Italian",   "price_range": "$$$", "vibe": "romantic"}},
    {"item_id": "i_004", "name": "Taco Libre",               "category": "Food",      "attributes": {"cuisine": "Mexican",   "price_range": "$",   "vibe": "lively"}},
    {"item_id": "i_005", "name": "The Vegan Corner",         "category": "Food",      "attributes": {"cuisine": "Vegan",     "price_range": "$$",  "vibe": "chill"}},
    {"item_id": "i_006", "name": "Burger Republic",          "category": "Food",      "attributes": {"cuisine": "American",  "price_range": "$",   "vibe": "casual"}},
    {"item_id": "i_007", "name": "Spice Garden",             "category": "Food",      "attributes": {"cuisine": "Indian",    "price_range": "$$",  "vibe": "warm"}},
    # Nightlife
    {"item_id": "i_101", "name": "The Rooftop Lounge",       "category": "Nightlife", "attributes": {"type": "bar",       "price_range": "$$",  "vibe": "chill"}},
    {"item_id": "i_102", "name": "Eclipse Nightclub",        "category": "Nightlife", "attributes": {"type": "club",      "price_range": "$$$", "vibe": "energetic"}},
    {"item_id": "i_103", "name": "The Jazz Cellar",          "category": "Nightlife", "attributes": {"type": "jazz bar",  "price_range": "$$",  "vibe": "sophisticated"}},
    {"item_id": "i_104", "name": "Brew & Board",             "category": "Nightlife", "attributes": {"type": "pub",       "price_range": "$",   "vibe": "casual"}},
    {"item_id": "i_105", "name": "Skyline Sky Bar",          "category": "Nightlife", "attributes": {"type": "bar",       "price_range": "$$$", "vibe": "upscale"}},
    # Movies
    {"item_id": "i_301", "name": "Interstellar",             "category": "Movies",    "attributes": {"genre": "Sci-fi",    "director": "Nolan",    "runtime_min": 169}},
    {"item_id": "i_302", "name": "The Grand Budapest Hotel", "category": "Movies",    "attributes": {"genre": "Comedy",    "director": "Anderson", "runtime_min": 99}},
    {"item_id": "i_303", "name": "Parasite",                 "category": "Movies",    "attributes": {"genre": "Thriller",  "director": "Bong",     "runtime_min": 132}},
    {"item_id": "i_304", "name": "Everything Everywhere",    "category": "Movies",    "attributes": {"genre": "Comedy",    "director": "Daniels",  "runtime_min": 139}},
    # Shopping
    {"item_id": "i_401", "name": "Central Market",           "category": "Shopping",  "attributes": {"type": "market",      "price_range": "$",   "vibe": "busy"}},
    {"item_id": "i_402", "name": "The Vintage Store",        "category": "Shopping",  "attributes": {"type": "boutique",    "price_range": "$$",  "vibe": "quirky"}},
    {"item_id": "i_403", "name": "Tech Haven",               "category": "Shopping",  "attributes": {"type": "electronics", "price_range": "$$$", "vibe": "modern"}},
    # Outdoors
    {"item_id": "i_501", "name": "Riverside Park Trail",     "category": "Outdoors",  "attributes": {"type": "hiking",  "difficulty": "easy",   "duration_hr": 2}},
    {"item_id": "i_502", "name": "Summit Ridge Hike",        "category": "Outdoors",  "attributes": {"type": "hiking",  "difficulty": "hard",   "duration_hr": 6}},
    {"item_id": "i_503", "name": "City Cycling Route",       "category": "Outdoors",  "attributes": {"type": "cycling", "difficulty": "medium", "duration_hr": 3}},
    # Arts
    {"item_id": "i_601", "name": "City Art Museum",          "category": "Arts",      "attributes": {"type": "museum",  "price_range": "$$",  "vibe": "cultural"}},
    {"item_id": "i_602", "name": "Improv Comedy Night",      "category": "Arts",      "attributes": {"type": "comedy",  "price_range": "$",   "vibe": "fun"}},
    {"item_id": "i_603", "name": "Symphony at the Hall",     "category": "Arts",      "attributes": {"type": "music",   "price_range": "$$$", "vibe": "formal"}},
]

# --- Nigerian items (bonus scoring!) ---
nigerian_items = [
    # Nigerian Food
    {"item_id": "ng_f01", "name": "Mama Put Kitchen",           "category": "Food", "attributes": {"cuisine": "Nigerian", "price_range": "$",   "vibe": "local", "region": "Lagos"}},
    {"item_id": "ng_f02", "name": "Amala Sky Restaurant",       "category": "Food", "attributes": {"cuisine": "Nigerian", "price_range": "$$",  "vibe": "upscale", "region": "Lagos"}},
    {"item_id": "ng_f03", "name": "Buka Hut",                   "category": "Food", "attributes": {"cuisine": "Nigerian", "price_range": "$",   "vibe": "casual", "region": "Abuja"}},
    {"item_id": "ng_f04", "name": "Suya Spot Express",          "category": "Food", "attributes": {"cuisine": "Nigerian-Grill", "price_range": "$", "vibe": "street food", "region": "Lagos"}},
    {"item_id": "ng_f05", "name": "Calabar Kitchen",            "category": "Food", "attributes": {"cuisine": "Nigerian-Calabar", "price_range": "$$", "vibe": "warm", "region": "Calabar"}},
    {"item_id": "ng_f06", "name": "Iya Basira Amala Joint",     "category": "Food", "attributes": {"cuisine": "Nigerian-Yoruba", "price_range": "$",  "vibe": "authentic", "region": "Lagos"}},
    {"item_id": "ng_f07", "name": "Jollof Republic",            "category": "Food", "attributes": {"cuisine": "Nigerian", "price_range": "$$",  "vibe": "modern", "region": "Lagos"}},
    {"item_id": "ng_f08", "name": "Kilimanjaro Restaurant",     "category": "Food", "attributes": {"cuisine": "Nigerian-FastFood", "price_range": "$$", "vibe": "family", "region": "Lagos"}},
    # Nigerian Nightlife
    {"item_id": "ng_n01", "name": "Quilox Nightclub",           "category": "Nightlife", "attributes": {"type": "club", "price_range": "$$$", "vibe": "VIP", "region": "Lagos"}},
    {"item_id": "ng_n02", "name": "Hard Rock Cafe Lagos",       "category": "Nightlife", "attributes": {"type": "bar", "price_range": "$$",  "vibe": "live music", "region": "Lagos"}},
    {"item_id": "ng_n03", "name": "Shiro Lagos",                "category": "Nightlife", "attributes": {"type": "lounge", "price_range": "$$$", "vibe": "upscale", "region": "Lagos"}},
    {"item_id": "ng_n04", "name": "Sky Lounge Abuja",           "category": "Nightlife", "attributes": {"type": "rooftop bar", "price_range": "$$", "vibe": "chill", "region": "Abuja"}},
    # Nigerian/African Books
    {"item_id": "ng_b01", "name": "Half of a Yellow Sun",       "category": "Books", "attributes": {"genre": "Historical Fiction", "author": "Chimamanda Ngozi Adichie", "avg_rating": 4.31, "genres": ["Historical Fiction", "African Literature"]}},
    {"item_id": "ng_b02", "name": "Things Fall Apart",          "category": "Books", "attributes": {"genre": "Classics", "author": "Chinua Achebe", "avg_rating": 3.72, "genres": ["Classics", "African Literature", "Fiction"]}},
    {"item_id": "ng_b03", "name": "Purple Hibiscus",            "category": "Books", "attributes": {"genre": "Fiction", "author": "Chimamanda Ngozi Adichie", "avg_rating": 4.08, "genres": ["Fiction", "African Literature"]}},
    {"item_id": "ng_b04", "name": "The Famished Road",          "category": "Books", "attributes": {"genre": "Magical Realism", "author": "Ben Okri", "avg_rating": 3.60, "genres": ["Magical Realism", "African Literature"]}},
    {"item_id": "ng_b05", "name": "Americanah",                 "category": "Books", "attributes": {"genre": "Contemporary Fiction", "author": "Chimamanda Ngozi Adichie", "avg_rating": 4.28, "genres": ["Fiction", "Contemporary", "African Literature"]}},
    {"item_id": "ng_b06", "name": "My Life in the Bush of Ghosts", "category": "Books", "attributes": {"genre": "Fiction", "author": "Amos Tutuola", "avg_rating": 3.78, "genres": ["Fiction", "African Literature", "Fantasy"]}},
    {"item_id": "ng_b07", "name": "Stay With Me",               "category": "Books", "attributes": {"genre": "Fiction", "author": "Ayobami Adebayo", "avg_rating": 3.88, "genres": ["Fiction", "African Literature", "Contemporary"]}},
    {"item_id": "ng_b08", "name": "An Orchestra of Minorities", "category": "Books", "attributes": {"genre": "Literary Fiction", "author": "Chigozie Obioma", "avg_rating": 3.72, "genres": ["Literary Fiction", "African Literature"]}},
    # Nollywood Movies
    {"item_id": "ng_m01", "name": "The Wedding Party",          "category": "Movies", "attributes": {"genre": "Comedy", "director": "Kemi Adetiba", "runtime_min": 120, "industry": "Nollywood"}},
    {"item_id": "ng_m02", "name": "Lionheart",                  "category": "Movies", "attributes": {"genre": "Drama", "director": "Genevieve Nnaji", "runtime_min": 95, "industry": "Nollywood"}},
    {"item_id": "ng_m03", "name": "King of Boys",               "category": "Movies", "attributes": {"genre": "Crime/Thriller", "director": "Kemi Adetiba", "runtime_min": 163, "industry": "Nollywood"}},
    {"item_id": "ng_m04", "name": "Citation",                   "category": "Movies", "attributes": {"genre": "Drama", "director": "Kunle Afolayan", "runtime_min": 120, "industry": "Nollywood"}},
    {"item_id": "ng_m05", "name": "Gangs of Lagos",             "category": "Movies", "attributes": {"genre": "Action/Drama", "director": "Jade Osiberu", "runtime_min": 120, "industry": "Nollywood"}},
    {"item_id": "ng_m06", "name": "A Sunday Affair",            "category": "Movies", "attributes": {"genre": "Romantic Comedy", "director": "Biodun Stephen", "runtime_min": 105, "industry": "Nollywood"}},
    # Nigerian Shopping
    {"item_id": "ng_s01", "name": "Computer Village Ikeja",     "category": "Shopping", "attributes": {"type": "electronics market", "price_range": "$", "vibe": "bustling", "region": "Lagos"}},
    {"item_id": "ng_s02", "name": "Lekki Arts & Crafts Market", "category": "Shopping", "attributes": {"type": "crafts market", "price_range": "$$", "vibe": "cultural", "region": "Lagos"}},
]

# Merge
full_catalog = catalog + multi_cat + nigerian_items
print(f"Final catalog: {len(full_catalog)} items")

cats = Counter(item["category"] for item in full_catalog)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

ng_count = sum(1 for item in full_catalog if item["item_id"].startswith("ng_"))
print(f"  Nigerian items: {ng_count}")

with open(CATALOG_PATH, "w", encoding="utf-8") as f:
    json.dump(full_catalog, f, indent=2)

size_kb = os.path.getsize(CATALOG_PATH) // 1024
print(f"catalog.json size: {size_kb} KB")
