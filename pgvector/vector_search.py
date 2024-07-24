import sys
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np

# Database connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost"
}


# Function to drop the table if it exists
def drop_table(cur):
    cur.execute("""
    DROP TABLE IF EXISTS items
    """)

# Function to create the table if it doesn't exist
def create_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(384)
    )
    """)

# Function to insert items
def insert_items(cur, items):
    insert_query = "INSERT INTO items (content, embedding) VALUES %s"
    execute_values(cur, insert_query, items)

# Function to perform similarity search
def similarity_search(cur, query_vector):
    search_query = """
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM items
    ORDER BY embedding <=> %s::vector
    LIMIT 1
    """
    cur.execute(search_query, (query_vector, query_vector))
    return cur.fetchone()

def main():
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Drop the table if it exists
        drop_table(cur)

        # Create the table if it doesn't exist
        create_table(cur)

        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Sample data
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A journey of a thousand miles begins with a single step",
            "To be or not to be, that is the question",
            "In the middle of difficulty lies opportunity"
        ]

        # Generate embeddings
        embeddings = model.encode(texts)

        # Prepare items for insertion
        items = [(text, embedding.tolist()) for text, embedding in zip(texts, embeddings)]

        # Insert items
        insert_items(cur, items)
        conn.commit()
        print(f"Inserted {len(texts)} items into the database.")

        # Perform a similarity search
        #query_text = "What is the meaning of life?"
        #query_text = "What did the fox jump?"
        #query_text = "How difficult is life?"
        query_text = "How many miles?"
        query_embedding = model.encode([query_text])[0].tolist()

        result = similarity_search(cur, query_embedding)

        if result:
            print(f"Most similar to '{query_text}':")
            print(f"Content: {result[0]}")
            print(f"Similarity: {result[1]:.4f}")
        else:
            print("No results found.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.commit()
            conn.close()

if __name__ == "__main__":
    main()
