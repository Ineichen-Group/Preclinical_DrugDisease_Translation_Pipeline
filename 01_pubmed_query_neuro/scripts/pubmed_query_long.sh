#!/bin/bash

# Loop through each query file
for file in ../data/pubmed_queries/nervous_system/cns_free_text_query_*.txt; do
    # Read the query from the file
    QUERY=$(cat "$file")

    # Extract the base name of the file without the extension
    BASENAME=$(basename "$file" .txt)

    # Run esearch and save the output to a file
    esearch -db pubmed -query "$QUERY" | efetch -format uid > "../data/pubmed_queries/results_pmids/${BASENAME}_$(date +%Y%m%d).txt"

    # Print a message indicating completion for this file
    echo "Processed $file and saved PMIDs to ${BASENAME}_$(date +%Y%m%d).txt"
done
