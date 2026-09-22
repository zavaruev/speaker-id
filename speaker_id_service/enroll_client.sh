#!/bin/bash
# ============================================================================
# Speaker ID — CLI enrollment client.
#
# Interactive alternative to the /enroll browser UI: records 3 samples of 8
# seconds each with arecord (ALSA), then uploads them to POST /enroll.
#
# IMPORTANT: /enroll is protected by an API key. This script sends it via the
# "X-API-Key" header. The key is taken from the SPEAKER_ID_API_KEY environment
# variable; if unset, you are prompted (input hidden). Earlier versions of
# this script did NOT send the key and always failed with HTTP 401.
#
# Usage:
#   export SPEAKER_ID_API_KEY="your-key"
#   ./enroll_client.sh
#
# Optional environment overrides:
#   SPEAKER_ID_SERVER   base URL of the service
#                       (default: http://192.168.22.102:8001)
#   DURATION / SAMPLES  recording length in seconds / number of samples
#
# Requirements: bash, curl, arecord (alsa-utils).
# ============================================================================

SERVER_URL="${SPEAKER_ID_SERVER:-http://192.168.22.102:8001}/enroll"
DURATION="${DURATION:-8}"
SAMPLES="${SAMPLES:-3}"

echo "========================================="
echo "Speaker ID Enrollment"
echo "========================================="

read -p "Enter your name (e.g. 'alexander'): " RAW_NAME

if [ -z "$RAW_NAME" ]; then
    echo "Error: Name cannot be empty!"
    exit 1
fi

# Normalize the name: lowercase everything, then capitalize the first letter.
# The server additionally enforces ^[a-zA-Z0-9_-]+$ after basename-stripping.
LOWER_NAME="${RAW_NAME,,}"
USER_NAME="${LOWER_NAME^}"

# Fixed prompts read aloud during recording; varied content/intonation gives
# the averaged embedding better coverage than repeating one sentence.
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
    FILE_PATH=$(mktemp "/tmp/${USER_NAME}_enroll_${i}_XXXXXX.wav")
    FILES+=("$FILE_PATH")

    echo "--- Sample $i of $SAMPLES ---"
    echo "Text: ${TEXTS[$i-1]}"
    echo ""
    read -p "Press [ENTER] to record sample $i..."

    echo "Recording sample $i! Read the text..."
    # 16 kHz mono signed 16-bit LE WAV — exactly what the server pipeline wants.
    arecord -f S16_LE -r 16000 -c 1 -d "$DURATION" "$FILE_PATH" > /dev/null 2>&1

    if [ ! -f "$FILE_PATH" ]; then
        echo "Error: Failed to record audio. Check your microphone."
        exit 1
    fi

    echo "Sample $i recorded"
    echo ""
done

# Resolve the API key up front so we fail before making the user record again:
# env var wins; otherwise ask quietly (no echo to screen/history).
API_KEY="${SPEAKER_ID_API_KEY:-}"
if [ -z "$API_KEY" ]; then
    read -s -p "API Key (X-API-Key): " API_KEY; echo ""
    if [ -z "$API_KEY" ]; then
        echo "Error: API key is required for enrollment."
        exit 1
    fi
fi

echo "Sending $SAMPLES samples for user '$USER_NAME' to server..."

CURL_ARGS=("-s" "-w" "\nHTTP_STATUS:%{http_code}" "-X" POST "$SERVER_URL" \
    "-H" "accept: application/json" \
    "-H" "X-API-Key: ${API_KEY}" \
    "-F" "user_id=${USER_NAME}")

for f in "${FILES[@]}"; do
    CURL_ARGS+=("-F" "files=@${f}")
done

RESPONSE=$(curl "${CURL_ARGS[@]}")

# Split the combined output into body and trailing HTTP_STATUS marker.
HTTP_STATUS=$(echo "$RESPONSE" | tr -d '\n' | sed -e 's/.*HTTP_STATUS://')
BODY=$(echo "$RESPONSE" | sed -e 's/HTTP_STATUS\:.*//g')

echo ""
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Success! Voice for user '$USER_NAME' enrolled from $SAMPLES samples."
    echo "Server response: $BODY"
else
    # 401 here means wrong/missing API key; 400 = validation, 413 = file too big.
    echo "Error. HTTP Status: $HTTP_STATUS"
    echo "Server response: $BODY"
fi

# Clean up local recordings regardless of outcome.
for f in "${FILES[@]}"; do
    rm -f "$f"
done
