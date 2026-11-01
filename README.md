# Own Data Search — Telegram-бот v2 (Haystack + Docling + Pinecone)

Модульная версия персонального помощника на базе урока `0806 PesonalAssistant`:  
все прежние функции плюс разбор файлов через **Docling** и сохранение чанков в **Pinecone**.

## Как закрыто домашнее задание

| # | Пункт ДЗ | Как реализовано |
|---|---|---|
| 1 | **Новая версия бота с модульной архитектурой** | Проект в корне репозитория. Логика разделена на `components/` (инфраструктура и сервисы), `pipelines/` (ingestion / generation) и `bot/` (Telegram). Точка входа — `main.py`. |
| 2 | **Интеграция Docling** | `pipelines/ingestion.py`: `DoclingConverter` в режиме `ExportType.DOC_CHUNKS` + `HybridChunker`. Поддерживаются PDF, DOCX и другие форматы Docling; файл скачивается из Telegram и разбирается на смысловые чанки. |
| 3 | **Сохранение в Pinecone** | После Docling чанки проходят `DocumentMetaEnricher` (meta: `file_name`, `chunk_index`, `page_number`, `user_id`) → `OpenAIDocumentEmbedder` → `DocumentWriter` → `PineconeDocumentStore` (`metric=cosine`). |
| 4 | **Резюме и ответы с учётом документов** | После успешной загрузки бот пишет «Готово…» и **одно** предложение-резюме (`components/summarizer.py`). На текстовые вопросы срабатывает `generation_pipeline`: `PineconeEmbeddingRetriever` подтягивает факты пользователя и чанки файлов → Agent/LLM отвечает по этому контексту. |

Проверка end-to-end: в Telegram загружался тестовый PDF, чанки записались в Pinecone, затем бот отвечал на вопросы по содержимому документа через retriever.

## Архитектура

```text
.
├── main.py                 # точка входа
├── components/
│   ├── config.py           # настройки из .env
│   ├── store.py            # Pinecone + embedders + retriever
│   ├── tools.py            # tools агента (кошки/собаки/погода/vision)
│   ├── memory.py           # факты о пользователе
│   ├── enricher.py         # meta для чанков Docling
│   ├── summarizer.py       # одно предложение-резюме файла
│   └── assistant.py        # фасад для бота
├── pipelines/
│   ├── ingestion.py        # Docling → enrich → embed → Pinecone
│   └── generation.py       # retrieve → Agent → ответ
└── bot/
    ├── handlers.py         # Telegram handlers
    └── messaging.py        # тексты и отправка сообщений
```

### Потоки

**Текст**

```text
Telegram → generation_pipeline
             ├─ PineconeEmbeddingRetriever (факты + чанки файлов)
             ├─ Agent + tools → ответ
             └─ extract facts → Pinecone
```

**Файл**

```text
Telegram document
  → «Файл получен…»
  → DoclingConverter (DOC_CHUNKS / HybridChunker)
  → meta: file_name, chunk_index, page_number, user_id
  → embeddings → Pinecone
  → «Готово…»
  → одно предложение-резюме
```

## Быстрый старт

```powershell
cd "c:\Users\User\Desktop\Course\0807 Own_Data_Search"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # затем заполнить ключи
python main.py
```

Нужны (только в локальном `.env`, не в git):  
`OPENAI_KEY`, `OPENAI_BASE_URL`, `PINICONE_TOKEN`, `PINECONE_INDEX_NAME`, `BOT_TOKEN`.  
Индекс Pinecone: dimension `1536`, metric `cosine`.

Образец переменных — `.env.example` (без секретов).

## Что умеет (как в v1 + файлы)

- диалог с персональной памятью (только факты о пользователе);
- tools: факт о кошках / собаках, картинка собаки + vision, погода Open-Meteo;
- команды `/start` `/help` `/memory` `/forget`;
- загрузка PDF/DOCX/… → Docling → Pinecone;
- вопросы по загруженным документам через retriever.

## Безопасность перед сдачей

В git **не** должны попадать:

- `.env` и любые API-/бот-токены;
- `venv/`;
- временные файлы из `tmp/` (скачанные из Telegram документы);
- пользовательские PDF/DOCX и прочие исходники документов;
- логи.

Используйте корневой `.gitignore`. Перед коммитом проверьте `git status`.
