#!/bin/bash

SERVER_URL="http://192.168.22.102:8001/enroll"
DURATION=8
SAMPLES=3

echo "========================================="
echo "Speaker ID Enrollment"
echo "========================================="

read -p "Enter your name (e.g. 'alexander'): " RAW_NAME

if [ -z "$RAW_NAME" ]; then
    echo "Error: Name cannot be empty!"
    exit 1
fi

LOWER_NAME="${RAW_NAME,,}"
USER_NAME="${LOWER_NAME^}"

TEXTS=(
    "Hello, computer! I am setting up my voice profile for the smart home system. This audio sample will help the neural network remember my voice."
    "The weather is great today, the sun is shining and the birds are singing outside. I hope the system recognizes my voice without errors even in a noisy room."
    "One, two, three, four, five, six, seven, eight, nine, ten. I am speaking with different intonation to make the sample as complete and high-quality as possible."
)

echo ""
echo "Recording $SAMPLES samples, $DURATION seconds each."
echo "Read the text aloud in your natural voice."
echo ""

FILES=()

for ((i=1; i<=SAMPLES; i++)); do
    FILE_PATH="/tmp/${USER_NAME}_enroll_${i}.wav"
    FILES+=("$FILE_PATH")

    echo "--- Sample $i of $SAMPLES ---"
    echo "Text: ${TEXTS[$i-1]}"
    echo ""
    read -p "Press [ENTER] to record sample $i..."

    echo "Recording sample $i! Read the text..."
    arecord -f S16_LE -r 16000 -c 1 -d $DURATION "$FILE_PATH" > /dev/null 2>&1

    if [ ! -f "$FILE_PATH" ]; then
        echo "Error: Failed to record audio. Check your microphone."
        exit 1
    fi

    echo "Sample $i recorded"
    echo ""
done

echo "Sending $SAMPLES samples for user '$USER_NAME' to server..."

CURL_ARGS=("-s" "-w" "\nHTTP_STATUS:%{http_code}" "-X" POST "$SERVER_URL" \
    "-H" "accept: application/json" \
    "-F" "user_id=${USER_NAME}")

for f in "${FILES[@]}"; do
    CURL_ARGS+=("-F" "files=@${f}")
done

RESPONSE=$(curl "${CURL_ARGS[@]}")

HTTP_STATUS=$(echo "$RESPONSE" | tr -d '\n' | sed -e 's/.*HTTP_STATUS://')
BODY=$(echo "$RESPONSE" | sed -e 's/HTTP_STATUS\:.*//g')

echo ""
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Success! Voice for user '$USER_NAME' enrolled from $SAMPLES samples."
    echo "Server response: $BODY"
else
    echo "Error. HTTP Status: $HTTP_STATUS"
    echo "Server response: $BODY"
fi

for f in "${FILES[@]}"; do
    rm -f "$f"
done
