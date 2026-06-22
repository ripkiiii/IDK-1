import json
import time
import random
from openai import OpenAI

API_KEY = "997733b646dd4a388c67def87ad0758d.jfOBGQDCWd8uwGgX"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4.7-flash"
OUTPUT_FILE = "idk1_instruct_1k.jsonl"
TARGET_PAIRS = 200
PAIRS_PER_CALL = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

TOPICS = [
    "pengetahuan umum tentang Indonesia (sejarah, geografi, budaya, tokoh nasional)",
    "kehidupan sehari-hari (masak, belanja, transportasi, keuangan pribadi)",
    "teknologi dan komputer (penjelasan konsep, troubleshooting, tips)",
    "menulis (surat, email formal, esai, ringkasan)",
    "matematika dan logika (soal cerita, perhitungan, penalaran)",
    "sains dan kesehatan (biologi, fisika, tips kesehatan)",
    "bahasa Indonesia (grammar, EYD, sinonim, penggunaan kata)",
    "coding dan pemrograman (Python, JavaScript, konsep dasar)",
    "bisnis dan kewirausahaan (tips usaha, pemasaran, manajemen)",
    "kreativitas (puisi pendek, cerita singkat, brainstorming ide)",
]

SYSTEM_PROMPT = """Kamu adalah generator dataset instruksi Bahasa Indonesia berkualitas tinggi.

Tugasmu: buat {n} pasang instruksi-respons dalam Bahasa Indonesia tentang topik: {topic}

Output HARUS berupa JSON array seperti ini:
[
  {{
    "instruction": "pertanyaan atau instruksi yang natural dalam bahasa Indonesia",
    "response": "jawaban lengkap dan akurat dalam bahasa Indonesia"
  }},
  ...
]

Aturan:
- Instruksi harus natural, seperti yang ditanyakan manusia sungguhan
- Respons harus informatif, jelas, dan tidak terlalu panjang (maks 300 kata)
- Variasikan tingkat kesulitan (mudah, sedang, susah)
- Gunakan bahasa Indonesia yang baik dan benar
- HANYA output JSON array, tidak ada teks lain"""


def generate_batch(topic: str, n: int = PAIRS_PER_CALL) -> list[dict]:
    prompt = SYSTEM_PROMPT.format(n=n, topic=topic)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    content = response.choices[0].message.content.strip()

    # strip markdown code block if ada
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    pairs = json.loads(content)
    return pairs


def to_chatml(instruction: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    pairs_done = 0
    calls_needed = TARGET_PAIRS // PAIRS_PER_CALL
    topic_cycle = [TOPICS[i % len(TOPICS)] for i in range(calls_needed)]
    random.shuffle(topic_cycle)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, topic in enumerate(topic_cycle):
            try:
                batch = generate_batch(topic)
                for pair in batch:
                    entry = to_chatml(pair["instruction"], pair["response"])
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    pairs_done += 1

                print(f"[{pairs_done}/{TARGET_PAIRS}] call {i+1}/{calls_needed} ✓ — {topic[:40]}")
                time.sleep(1.5)  # hindari rate limit

            except Exception as e:
                print(f"[ERROR] call {i+1}: {e} — skip, lanjut")
                time.sleep(3)

    print(f"\nDone! {pairs_done} pairs saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
