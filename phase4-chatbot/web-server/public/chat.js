// ============================================
// CHAT.JS — Browser-side JavaScript
// Handles: sending messages, showing replies,
//          typing indicators, server status
// ============================================

// ── State
let isWaiting = false   // prevent sending while waiting for reply

// ── DOM references
const chatWindow = document.getElementById('chat-window')
const input = document.getElementById('user-input')
const sendBtn = document.getElementById('send-btn')
const statusDot = document.getElementById('status-dot')
const statusText = document.getElementById('status-text')
const charCount = document.getElementById('char-count')
const tempSlider = document.getElementById('temp-slider')
const tempVal = document.getElementById('temp-val')
const suggestions = document.getElementById('suggestions')


// ============================================
// STARTUP
// ============================================
window.addEventListener('load', () => {
    // Set welcome message timestamp
    document.getElementById('welcome-time').textContent = getTime()

    // Check if servers are online
    checkHealth()

    // Focus input
    input.focus()
})


// ============================================
// HEALTH CHECK — is the LLM server running?
// ============================================
async function checkHealth() {
    try {
        const res = await fetch('/health')
        const data = await res.json()

        if (data.llmServer && data.llmServer.status === 'ok') {
            setStatus('online', 'LLM Online')
        } else {
            setStatus('offline', 'LLM Offline — run python app.py')
        }
    } catch {
        setStatus('offline', 'Server unreachable')
    }
}

function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state
    statusText.textContent = text
}


// ============================================
// SEND MESSAGE
// ============================================
async function sendMessage() {
    const text = input.value.trim()
    if (!text || isWaiting) return

    // Hide suggestions after first message
    suggestions.style.display = 'none'

    // Show user's message
    appendMessage('user', text)
    input.value = ''
    charCount.textContent = '0 / 200'
    autoResize()

    // Show typing indicator
    const typingId = showTyping()
    setWaiting(true)

    try {
        // Send to Node.js server
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                temperature: getTemperature(),
                n_chars: 150
            })
        })

        const data = await res.json()
        removeTyping(typingId)

        if (res.ok) {
            appendMessage('bot', data.response)
            setStatus('online', 'LLM Online')
        } else {
            appendError(data.error || 'Something went wrong.')
            setStatus('offline', 'Error')
        }

    } catch (err) {
        removeTyping(typingId)
        appendError('Cannot reach the server. Is Node.js running?')
        setStatus('offline', 'Disconnected')
    }

    setWaiting(false)
    input.focus()
}


// ============================================
// APPEND MESSAGES TO CHAT
// ============================================
function appendMessage(role, text) {
    const isBot = role === 'bot'
    const row = document.createElement('div')
    row.className = `msg-row ${isBot ? 'bot-row' : 'user-row'}`

    // Clean up text — trim leading/trailing whitespace and newlines
    const cleanText = text.replace(/^\s+/, '').replace(/\s+$/, '') || '...'

    row.innerHTML = `
        <div class="avatar ${isBot ? 'bot-avatar' : 'user-avatar'}">
            ${isBot ? 'AI' : 'YOU'}
        </div>
        <div class="bubble ${isBot ? 'bot-bubble' : 'user-bubble'}">
            <div class="bubble-header">
                ${isBot ? 'StockAI' : 'You'}
                <span class="bubble-time">${getTime()}</span>
            </div>
            <div class="bubble-body">${escapeHTML(cleanText)}</div>
        </div>
    `

    chatWindow.appendChild(row)
    scrollToBottom()
}

function appendError(message) {
    const row = document.createElement('div')
    row.className = 'msg-row bot-row'
    row.innerHTML = `
        <div class="avatar bot-avatar">AI</div>
        <div class="error-bubble">${escapeHTML(message)}</div>
    `
    chatWindow.appendChild(row)
    scrollToBottom()
}


// ============================================
// TYPING INDICATOR
// ============================================
function showTyping() {
    const id = 'typing-' + Date.now()
    const row = document.createElement('div')
    row.className = 'msg-row bot-row'
    row.id = id
    row.innerHTML = `
        <div class="avatar bot-avatar">AI</div>
        <div class="bubble bot-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `
    chatWindow.appendChild(row)
    scrollToBottom()
    return id
}

function removeTyping(id) {
    const el = document.getElementById(id)
    if (el) el.remove()
}


// ============================================
// UTILITY FUNCTIONS
// ============================================
function getTemperature() {
    return parseFloat(tempSlider.value) / 10
}

function getTime() {
    return new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    })
}

function scrollToBottom() {
    const main = document.querySelector('main')
    main.scrollTop = main.scrollHeight
}

function setWaiting(state) {
    isWaiting = state
    sendBtn.disabled = state
    input.disabled = state
}

function escapeHTML(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/\n/g, '<br>')
}

function autoResize() {
    input.style.height = 'auto'
    input.style.height = Math.min(input.scrollHeight, 100) + 'px'
}

function usePrompt(text) {
    input.value = text
    charCount.textContent = `${text.length} / 200`
    input.focus()
    autoResize()
}


// ============================================
// EVENT LISTENERS
// ============================================

// Send on Enter (Shift+Enter = new line)
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        sendMessage()
    }
})

// Character count + auto-resize
input.addEventListener('input', () => {
    charCount.textContent = `${input.value.length} / 200`
    autoResize()
})

// Temperature slider display
tempSlider.addEventListener('input', () => {
    tempVal.textContent = (tempSlider.value / 10).toFixed(1)
})

// Periodically re-check server health
setInterval(checkHealth, 30000)