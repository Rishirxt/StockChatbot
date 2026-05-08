let isWaiting = false
let isWidgetOpen = true

const body = document.body
const chatWindow = document.getElementById('chat-window')
const chatScroll = document.getElementById('chat-scroll')
const input = document.getElementById('user-input')
const sendBtn = document.getElementById('send-btn')
const statusDot = document.getElementById('status-dot')
const statusText = document.getElementById('status-text')
const charCount = document.getElementById('char-count')
const tempSlider = document.getElementById('temp-slider')
const tempVal = document.getElementById('temp-val')
const suggestions = document.getElementById('suggestions')
const launcher = document.getElementById('chat-launcher')
const toggleWidgetBtn = document.getElementById('toggle-widget')
const openChatBtn = document.getElementById('open-chat-btn')

window.addEventListener('load', () => {
    document.getElementById('welcome-time').textContent = getTime()
    checkHealth()
    updateWidgetState(true)
    input.focus()
})

async function checkHealth() {
    try {
        const res = await fetch('/health')
        const data = await res.json()

        if (data.llmServer && data.llmServer.status === 'ok') {
            setStatus('online', 'AI online')
        } else {
            setStatus('offline', 'LLM offline')
        }
    } catch {
        setStatus('offline', 'Server unreachable')
    }
}

function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state
    statusText.textContent = text
}

async function sendMessage() {
    const text = input.value.trim()
    if (!text || isWaiting) return

    suggestions.style.display = 'none'
    appendMessage('user', text)
    input.value = ''
    charCount.textContent = '0 / 200'
    autoResize()

    const typingId = showTyping()
    setWaiting(true)

    try {
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
            setStatus('online', 'AI online')
        } else {
            appendError(data.error || 'Something went wrong.')
            setStatus('offline', 'Reply failed')
        }
    } catch {
        removeTyping(typingId)
        appendError('Cannot reach the server. Start the Node.js and Python servers first.')
        setStatus('offline', 'Disconnected')
    }

    setWaiting(false)
    input.focus()
}

function appendMessage(role, text) {
    const isBot = role === 'bot'
    const row = document.createElement('div')
    row.className = `msg-row ${isBot ? 'bot-row' : 'user-row'}`

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
    chatScroll.scrollTop = chatScroll.scrollHeight
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
    openWidget()
    input.value = text
    charCount.textContent = `${text.length} / 200`
    input.focus()
    autoResize()
}

function updateWidgetState(open) {
    isWidgetOpen = open
    body.classList.toggle('widget-open', open)
    toggleWidgetBtn.setAttribute('aria-label', open ? 'Minimize chat' : 'Open chat')

    if (open) {
        setTimeout(() => {
            input.focus()
            scrollToBottom()
        }, 120)
    }
}

function openWidget() {
    updateWidgetState(true)
}

function toggleWidget() {
    updateWidgetState(!isWidgetOpen)
}

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        sendMessage()
    }
})

input.addEventListener('input', () => {
    charCount.textContent = `${input.value.length} / 200`
    autoResize()
})

tempSlider.addEventListener('input', () => {
    tempVal.textContent = (tempSlider.value / 10).toFixed(1)
})

launcher.addEventListener('click', openWidget)
toggleWidgetBtn.addEventListener('click', toggleWidget)
openChatBtn.addEventListener('click', openWidget)

setInterval(checkHealth, 30000)
