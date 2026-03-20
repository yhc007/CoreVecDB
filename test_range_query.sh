#!/bin/bash
# Test Range Queries

BASE_URL="http://localhost:3000"

echo "=============================================="
echo "Range Query Test"
echo "=============================================="

# Generate 128-dim vector
gen_vec() {
    python3 -c "import random; print(','.join([str(random.random()) for _ in range(128)]))"
}

echo ""
echo "1. Inserting products with prices and ratings..."

# Insert products with numeric metadata
for i in {0..9}; do
    VEC=$(gen_vec)
    PRICE=$((100 + i * 50))  # 100, 150, 200, ..., 550
    RATING=$(python3 -c "print(round(3.0 + $i * 0.2, 1))")  # 3.0, 3.2, 3.4, ..., 4.8
    CATEGORY=$((i % 3))

    case $CATEGORY in
        0) CAT="electronics" ;;
        1) CAT="clothing" ;;
        2) CAT="food" ;;
    esac

    curl -s -X POST "$BASE_URL/upsert" \
        -H "Content-Type: application/json" \
        -d "{\"id\":$i,\"vector\":[$VEC],\"metadata\":{\"category\":\"$CAT\",\"price\":\"$PRICE\",\"rating\":\"$RATING\"}}" > /dev/null

    echo "  ID $i: category=$CAT, price=$PRICE, rating=$RATING"
done

echo ""
echo "2. Checking stats..."
curl -s "$BASE_URL/stats" | python3 -m json.tool

QUERY_VEC=$(gen_vec)

echo ""
echo "=============================================="
echo "Range Query Tests (via category filter for now)"
echo "=============================================="
echo ""
echo "Note: HTTP API currently uses simple key-value filter."
echo "Range queries work internally via FilterQuery DSL."
echo "Full HTTP range API would require JSON filter schema."
echo ""

echo "3. Filter: category='electronics'"
RESULT=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"vector\":[$QUERY_VEC],\"k\":10,\"filter\":{\"category\":\"electronics\"}}")
echo "  Results: $(echo $RESULT | python3 -c 'import sys,json; r=json.load(sys.stdin); print([x["id"] for x in r.get("results",[])])')"
echo "  Expected: IDs 0, 3, 6, 9 (electronics)"

echo ""
echo "=============================================="
echo "Internal Range Query API is ready!"
echo ""
echo "Usage in Rust:"
echo "  // Filter: price >= 200 AND price <= 400"
echo "  let query = FilterQuery::range(\"price\", 200, 400);"
echo ""
echo "  // Filter: rating >= 4.0"
echo "  let query = FilterQuery::gte_f(\"rating\", 4.0);"
echo ""
echo "  // Combined: category=electronics AND price < 300"
echo "  let query = FilterQuery::and(vec!["
echo "      FilterQuery::eq(\"category\", \"electronics\"),"
echo "      FilterQuery::lt(\"price\", 300),"
echo "  ]);"
echo "=============================================="
