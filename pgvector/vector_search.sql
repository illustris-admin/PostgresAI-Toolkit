-- Create and Connect To PostgreSQL Database
DROP DATABASE IF EXISTS pgvector;
CREATE DATABASE pgvector;
\c pgvector

-- Enable the vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the items table if exist
DROP TABLE IF EXISTS items;

-- Create the items table if it doesn't exist
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)
);

-- Function to simulate embedding generation (in real-world scenarios, this would be done in the application layer)
-- Note: This function should ideally use a pre-trained model like BERT or Sentence-BERT.
-- For demonstration purposes, we'll keep the random generation.
DROP FUNCTION IF EXISTS generate_mock_embedding_v2;
CREATE OR REPLACE FUNCTION generate_mock_embedding_v2(text TEXT) RETURNS vector(384) AS $$
DECLARE
    v vector(384);
BEGIN
    -- Generate a random 384-dimensional vector for demonstration purposes
    SELECT array_agg(random())::vector(384) INTO v
    FROM generate_series(1, 384);
    RETURN v;
END;
$$


LANGUAGE plpgsql;

-- Function to insert items
DROP FUNCTION IF EXISTS insert_items_v2;
CREATE OR REPLACE FUNCTION insert_items_v2() RETURNS integer AS $$
DECLARE
    texts TEXT[] := ARRAY[
        'The quick brown fox jumps over the lazy dog',
        'A journey of a thousand miles begins with a single step',
        'To be or not to be, that is the question',
        'In the middle of difficulty lies opportunity'
    ];
    t TEXT;
    inserted_count integer := 0;
BEGIN
    FOREACH t IN ARRAY texts
    LOOP
        RAISE NOTICE 'Inserting text: %', t;
        INSERT INTO items (content, embedding) VALUES (t, generate_mock_embedding_v2(t));
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RAISE NOTICE 'Total inserted: %', inserted_count;
    RETURN inserted_count;
END;
$$


LANGUAGE plpgsql;

-- Function to perform similarity search
DROP FUNCTION IF EXISTS similarity_search_v2;
CREATE OR REPLACE FUNCTION similarity_search_v2(query_text TEXT) RETURNS TABLE (content TEXT, similarity FLOAT) AS $$
DECLARE
    query_embedding vector(384);
BEGIN
    -- Generate query embedding based on the query text
    query_embedding := generate_mock_embedding_v2(query_text);
    
    RETURN QUERY
    SELECT i.content, 1 - (i.embedding <=> query_embedding) AS similarity
    FROM items i
    ORDER BY i.embedding <=> query_embedding
    LIMIT 5;
END;
$$


LANGUAGE plpgsql;

-- Procedure to run the entire process with a modifiable query text
DROP PROCEDURE IF EXISTS run_vector_demo_v2;
CREATE OR REPLACE PROCEDURE run_vector_demo_v2(query_text TEXT) AS $$
DECLARE
    inserted_count integer;
    result RECORD;
BEGIN
    -- Insert items
    BEGIN
        inserted_count := insert_items_v2();
        RAISE NOTICE 'Inserted % items into the database.', inserted_count;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Error inserting items: %', SQLERRM;
    END;

    -- Print the original query text
    RAISE NOTICE 'Original query text: %', query_text;

    -- Perform a similarity search
    BEGIN
        FOR result IN SELECT * FROM similarity_search_v2(query_text)
        LOOP
            RAISE NOTICE 'Most similar content: %', result.content;
            RAISE NOTICE 'Similarity: %', result.similarity;
        END LOOP;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Error performing similarity search: %', SQLERRM;
    END;

EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'Error in run_vector_demo_v2: %', SQLERRM;
END;
$$


LANGUAGE plpgsql;

-- Execute the demo with a modifiable query text
DO $$
DECLARE
--    query_text TEXT := 'What did the fox jump?';
--    query_text TEXT := 'brown fox';
--     query_text TEXT := 'What is the meaning of life?';
--     query_text TEXT := 'How many miles?';
     query_text TEXT := 'thousand miles';
BEGIN
    CALL run_vector_demo_v2(query_text);
END;
$$
;
