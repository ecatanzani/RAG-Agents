# Product Checker Agent 🤖📦

Un agente conversazionale intelligente (chatbot) progettato per verificare istantaneamente se un determinato prodotto è incluso, approvato o censito all'interno di una specifica lista di riferimento.

L'agente è in grado di comprendere il linguaggio naturale, gestire variazioni nei nomi dei prodotti, errori di battitura o sinonimi, fornendo una risposta immediata e motivata all'utente.

---

## 🎯 Scopo dell'Agente

Il chatbot nasce per automatizzare il controllo di conformità o disponibilità. Riceve in input il nome o il codice di un prodotto da parte dell'utente e interroga la lista di riferimento per determinare se è presente o meno, spiegandone eventualmente i criteri.

**Esempi di applicazione:**
*   Verifica l'inclusione di prodotti in incentivi statali o bonus fiscali.
*   Controllo di conformità a cataloghi aziendali o whitelist di fornitori.
*   Verifica della presenza di allergeni o ingredienti specifici in una lista.

## 🚀 Funzionalità Principali

*   **Riconoscimento Flessibile (Fuzzy Matching):** Capisce il prodotto anche se l'utente commette errori di ortografia o usa sinonimi.
*   **Query Multi-Parametro:** Permette la ricerca per nome commerciale, marca o codice identificativo (EAN/SKU).
*   **Risposte Motivate:** Non si limita a un "Sì/No", ma spiega *perché* il prodotto è incluso o escluso, citando la categoria o la regola di riferimento.
*   **Aggiornamento Dinamico:** La lista dei prodotti può essere aggiornata in tempo reale senza dover riaddestrare l'agente.

## 🛠️ Stack Tecnologico

*   **LLM Core:** [es. OpenAI GPT-4o-mini / Anthropic Claude 3 Haiku]
*   **Framework Agente:** [es. LangChain / CrewAI / Python native]
*   **Storage Lista:** [es. File CSV / JSON / Database PostgreSQL / Vector DB]

## 📦 Installazione e Setup

### Prerequisiti
*   Python 3.10+
*   Chiave API per il modello LLM scelto (`.env`)
