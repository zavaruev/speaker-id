#!/bin/bash

SERVER_URL="http://192.168.22.102:8001/enroll"
DURATION=8
SAMPLES=3

echo "========================================="
echo "🎤 Регистрация нового голоса в Speaker ID"
echo "========================================="

read -p "Введите ваше имя (на английском, например 'alexander'): " RAW_NAME

if [ -z "$RAW_NAME" ]; then
    echo "❌ Ошибка: Имя не может быть пустым!"
    exit 1
fi

LOWER_NAME="${RAW_NAME,,}"
USER_NAME="${LOWER_NAME^}"

TEXTS=(
    "Привет, компьютер! Я настраиваю свой голосовой профиль для системы умного дома. Этот образец звука поможет нейросети запомнить мой голос."
    "Сегодня отличная погода, за окном светит солнце и поют птицы. Надеюсь, система распознает мой голос без ошибок даже в шумной комнате."
    "Раз, два, три, четыре, пять, шесть, семь, восемь, девять, десять. Я говорю с разной интонацией, чтобы сэмпл получился максимально полным и качественным."
)

echo ""
echo "Будет записано $SAMPLES сэмплов по $DURATION секунд каждый."
echo "Прочитайте вслух предложенный текст естественным голосом."
echo ""

FILES=()

for ((i=1; i<=SAMPLES; i++)); do
    FILE_PATH="/tmp/${USER_NAME}_enroll_${i}.wav"
    FILES+=("$FILE_PATH")

    echo "--- Сэмпл $i из $SAMPLES ---"
    echo "Текст: ${TEXTS[$i-1]}"
    echo ""
    read -p "Нажмите [ENTER] для записи сэмпла $i..."

    echo "🔴 ЗАПИСЬ $i! Читайте текст..."
    arecord -f S16_LE -r 16000 -c 1 -d $DURATION "$FILE_PATH" > /dev/null 2>&1

    if [ ! -f "$FILE_PATH" ]; then
        echo "❌ Ошибка: Не удалось записать аудио. Проверь микрофон."
        exit 1
    fi

    echo "✅ Сэмпл $i записан"
    echo ""
done

echo "Отправка $SAMPLES сэмплов пользователя '$USER_NAME' на сервер..."

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
    echo "🎉 Успех! Голос пользователя '$USER_NAME' зарегистрирован по $SAMPLES сэмплам."
    echo "Ответ сервера: $BODY"
else
    echo "❌ Ошибка. HTTP Статус: $HTTP_STATUS"
    echo "Ответ сервера: $BODY"
fi

for f in "${FILES[@]}"; do
    rm -f "$f"
done
