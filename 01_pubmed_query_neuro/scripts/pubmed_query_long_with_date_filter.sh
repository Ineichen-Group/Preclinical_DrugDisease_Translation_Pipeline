#!/bin/bash

DATE_FILTER='("2024/12/01"[dp] : "2025/12/31"[dp])'
OUTPUT_DIR="../data/pubmed_queries/results_pmids/update_2025"

mkdir -p "$OUTPUT_DIR"

for file in ../data/pubmed_queries/nervous_system/cns_free_text_query_*.txt; do
    # Read the query and remove possible Windows carriage returns
    QUERY=$(tr -d '\r' < "$file")

    # Combine the original query with the date restriction
    FULL_QUERY="($QUERY) AND $DATE_FILTER"

    BASENAME=$(basename "$file" .txt)
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_$(date +%Y%m%d).txt"

    esearch -db pubmed -query "$FULL_QUERY" \
        | efetch -format uid \
        > "$OUTPUT_FILE"

    echo "Processed $file and saved PMIDs to $OUTPUT_FILE"
done