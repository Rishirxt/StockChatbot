// ============================================
// SERVER.JS — Node.js Express Web Server
// Serves the chat UI and forwards messages to Python
// Run with: node server.js
// Listens on: http://localhost:3000
// ============================================

const express = require('express')
const path = require('path')

const app = express()
const PORT = 3000
const LLM_URL = 'http://localhost:5000'   // Python Flask server

// ── Middleware
// Parse incoming JSON request bodies
app.use(express.json())

// Serve static files (index.html, style.css, chat.js) from public/
app.use(express.static(path.join(__dirname, 'public')))


// ============================================
// ROUTE — Health check
// ============================================
app.get('/health', async (req, res) => {
    try {
        // Check if Python LLM server is running
        const response = await fetch(`${LLM_URL}/health`)
        const data = await response.json()
        res.json({ nodeServer: 'ok', llmServer: data })
    } catch (err) {
        res.status(503).json({
            nodeServer: 'ok',
            llmServer: 'unreachable — is python app.py running?'
        })
    }
})


// ============================================
// ROUTE — Chat endpoint
// Browser sends message → we forward to Python LLM → return response
// ============================================
app.post('/chat', async (req, res) => {
    const { message, temperature, n_chars } = req.body

    // Validate input
    if (!message || message.trim() === '') {
        return res.status(400).json({ error: 'Message cannot be empty' })
    }

    console.log(`[chat] User: "${message}"`)

    try {
        // Forward message to Python LLM server
        const llmResponse = await fetch(`${LLM_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                temperature: temperature || 1.0,
                n_chars: n_chars || 120
            })
        })

        if (!llmResponse.ok) {
            throw new Error(`LLM server error: ${llmResponse.status}`)
        }

        const data = await llmResponse.json()
        console.log(`[chat] LLM: "${data.response.slice(0, 50)}..."`)

        // Send back to browser
        res.json({
            response: data.response,
            timestamp: new Date().toISOString()
        })

    } catch (err) {
        console.error('[chat] Error:', err.message)

        // If Python server is down, tell the user clearly
        if (err.message.includes('fetch')) {
            return res.status(503).json({
                error: 'LLM server is not running. Please start python app.py first.'
            })
        }

        res.status(500).json({ error: 'Something went wrong. Try again.' })
    }
})


// ============================================
// START SERVER
// ============================================
app.listen(PORT, () => {
    console.log(`\nStock Chatbot running at http://localhost:${PORT}`)
    console.log('Make sure Python LLM server is also running:')
    console.log('  cd llm-server && python app.py\n')
})