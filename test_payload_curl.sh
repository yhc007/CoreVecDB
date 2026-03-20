#!/bin/bash
# Test Payload Index with curl

BASE_URL="http://localhost:3000"

echo "=============================================="
echo "Payload Index Test (curl)"
echo "=============================================="

# Generate 128-dim vector string
gen_vec() {
    python3 -c "import random; print(','.join([str(random.random()) for _ in range(128)]))"
}

echo ""
echo "1. Inserting test vectors with metadata..."

# Insert vectors with indexed metadata
for i in {0..9}; do
    VEC=$(gen_vec)
    CATEGORY=$((i % 3))

    case $CATEGORY in
        0) CAT="electronics" ;;
        1) CAT="books" ;;
        2) CAT="clothing" ;;
    esac

    case $CATEGORY in
        0) STATUS="available" ;;
        1) STATUS="sold" ;;
        2) STATUS="pending" ;;
    esac

    curl -s -X POST "$BASE_URL/upsert" \
        -H "Content-Type: application/json" \
        -d "{\"id\":$i,\"vector\":[$VEC],\"metadata\":{\"category\":\"$CAT\",\"status\":\"$STATUS\",\"name\":\"product_$i\"}}" > /dev/null

    echo "  Inserted $i: category=$CAT, status=$STATUS"
done

echo ""
echo "2. Checking stats..."
curl -s "$BASE_URL/stats" | python3 -m json.tool

# Store query vector for searches
QUERY_VEC=$(gen_vec)

echo ""
echo "3. Search WITHOUT filter (top 5)..."
RESULT=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"vector\":[$QUERY_VEC],\"k\":5,\"filter\":{}}")
echo "  Results: $(echo $RESULT | python3 -c 'import sys,json; r=json.load(sys.stdin); print([x["id"] for x in r.get("results",[])])')"

echo ""
echo "4. Search WITH filter: category='electronics' (indexed field)"
RESULT=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"vector\":[$QUERY_VEC],\"k\":5,\"filter\":{\"category\":\"electronics\"}}")
echo "  Results: $(echo $RESULT | python3 -c 'import sys,json; r=json.load(sys.stdin); print([x["id"] for x in r.get("results",[])])')"
echo "  Expected: should only contain IDs 0, 3, 6, 9 (category=electronics)"

echo ""
echo "5. Search WITH filter: category='books' AND status='sold'"
RESULT=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"vector\":[$QUERY_VEC],\"k\":5,\"filter\":{\"category\":\"books\",\"status\":\"sold\"}}")
echo "  Results: $(echo $RESULT | python3 -c 'import sys,json; r=json.load(sys.stdin); print([x["id"] for x in r.get("results",[])])')"
echo "  Expected: should only contain IDs 1, 4, 7 (category=books AND status=sold)"

echo ""
echo "6. Search WITH filter: name='product_5' (NOT indexed - uses post-filter)"
RESULT=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"vector\":[$QUERY_VEC],\"k\":10,\"filter\":{\"name\":\"product_5\"}}")
echo "  Results: $(echo $RESULT | python3 -c 'import sys,json; r=json.load(sys.stdin); print([x["id"] for x in r.get("results",[])])')"
echo "  Expected: should only contain ID 5"

echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
