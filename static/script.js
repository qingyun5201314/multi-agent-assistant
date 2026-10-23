const API_BASE = '';

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadKnowledgeBtn = document.getElementById('loadKnowledge');
const knowledgeStats = document.getElementById('knowledgeStats');
const maxStepsInput = document.getElementById('maxSteps');
const sessionList = document.getElementById('sessionList');
const newChatBtn = document.getElementById('newChatBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const micBtn = document.getElementById('micBtn');
const speakerBtn = document.getElementById('speakerBtn');

let currentSessionId = null;
let isSpeakerOn = true; // 语音播报开关，默认开启
let currentAudio = null;

function generateSessionId() {
    return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// ==================== 语音播报逻辑 ====================
if (speakerBtn) {
    speakerBtn.addEventListener('click', () => {
        isSpeakerOn = !isSpeakerOn;
        speakerBtn.textContent = isSpeakerOn ? '🔊' : '🔇';
        if (!isSpeakerOn && currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
    });
}

async function playTTS(text) {
    if (!isSpeakerOn) return;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    try {
        // 去除 Markdown 和长换行，防止读出奇怪符号，并截取前200字
        const cleanText = text.replace(/[*#`]/g, '').replace(/\n/g, '，').slice(0, 200);
        const response = await fetch(`${API_BASE}/tts?text=${encodeURIComponent(cleanText)}`);
        if (!response.ok) throw new Error('TTS 请求失败');
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        currentAudio = new Audio(audioUrl);
        currentAudio.play();
    } catch (error) { console.error('语音播报失败:', error); }
}

// ==================== 语音输入逻辑 ====================
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

if (micBtn) {
    micBtn.addEventListener('mousedown', startRecording);
    micBtn.addEventListener('mouseup', stopRecording);
    micBtn.addEventListener('mouseleave', stopRecording);
}

async function startRecording() {
    if (isRecording) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'record.webm');

            micBtn.textContent = '⏳';
            micBtn.classList.remove('recording');
            isRecording = false;

            try {
                const response = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success && data.text) {
                    userInput.value = data.text;
                    userInput.focus();
                } else {
                    alert('语音识别失败: ' + data.error);
                }
            } catch (error) {
                alert('语音识别网络请求失败');
            } finally {
                micBtn.textContent = '🎤';
                stream.getTracks().forEach(track => track.stop());
            }
        };

        mediaRecorder.start();
        isRecording = true;
        micBtn.textContent = '🟥';
        micBtn.classList.add('recording');
    } catch (error) {
        console.error("无法访问麦克风:", error);
        alert("无法访问麦克风，请确保浏览器允许录音权限。");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
    }
}

// ==================== 会话管理 ====================
async function loadSessionList() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        const data = await response.json();
        sessionList.innerHTML = '';
        if (data.sessions.length === 0) {
            sessionList.innerHTML = '<div class="empty-hint">暂无对话</div>';
            return;
        }
        data.sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = 'session-item';
            if (session.session_id === currentSessionId) item.classList.add('active');
            item.innerHTML = `
                <div class="session-title">${session.title || '新对话'}</div>
                <button class="session-delete" title="删除">×</button>
            `;
            item.querySelector('.session-title').addEventListener('click', () => switchSession(session.session_id));
            item.querySelector('.session-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('确定删除这个对话吗？')) deleteSession(session.session_id);
            });
            sessionList.appendChild(item);
        });
    } catch (error) { console.error('加载会话列表失败:', error); }
}

async function switchSession(sessionId) {
    currentSessionId = sessionId;
    try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
        const data = await response.json();
        chatMessages.innerHTML = '';
        if (data.messages.length === 0) {
            chatMessages.innerHTML = `<div class="welcome"><h2>👋 新对话</h2><p>继续提问吧！</p></div>`;
        } else {
            data.messages.forEach(msg => addMessage(msg.role, msg.content, false, null, false));
        }
        loadSessionList();
    } catch (error) { console.error('加载会话消息失败:', error); }
}

async function newChat() {
    currentSessionId = generateSessionId();
    chatMessages.innerHTML = `<div class="welcome"><h2>👋 新对话</h2><p>开始提问吧！</p></div>`;
    await loadSessionList();
}

async function deleteSession(sessionId) {
    try {
        await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' });
        if (sessionId === currentSessionId) await newChat();
        else await loadSessionList();
    } catch (error) { console.error('删除失败:', error); }
}

// ==================== 发送消息（流式+语音） ====================
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;
    if (!currentSessionId) currentSessionId = generateSessionId();
    
    userInput.value = '';
    sendBtn.disabled = true;
    sendBtn.textContent = '思考中';
    
    const welcome = chatMessages.querySelector('.welcome');
    if (welcome) welcome.remove();
    
    addMessage('user', query, false, null, true);
    
    const msgEl = document.createElement('div');
    msgEl.className = `message assistant`;
    msgEl.innerHTML = '<span class="loading">🤔 Agent 正在思考...</span>';
    chatMessages.appendChild(msgEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    let fullText = "";
    let isFirstText = true;
    let textContainer = null;

    const maxSteps = parseInt(maxStepsInput.value) || 5;
    const url = `${API_BASE}/chat_stream?query=${encodeURIComponent(query)}&session_id=${currentSessionId}&max_steps=${maxSteps}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'status') {
                msgEl.innerHTML = `<span class="loading">🔍 ${data.content}</span>`;
            } 
            else if (data.type === 'title') {
                loadSessionList();
            } 
            else if (data.type === 'text') {
                if (isFirstText) {
                    msgEl.innerHTML = ''; 
                    textContainer = document.createElement('div');
                    textContainer.style.whiteSpace = 'pre-wrap';
                    msgEl.appendChild(textContainer);
                    isFirstText = false;
                }
                fullText += data.content;
                textContainer.textContent = fullText;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } 
            else if (data.type === 'final') {
                if (data.steps && data.steps.length > 0) {
                    renderSteps(msgEl, data.steps);
                }
                loadSessionList();
                eventSource.close(); 
                
                // 触发语音播报
                if (fullText) playTTS(fullText);
            } 
            else if (data.type === 'error') {
                msgEl.textContent = `❌ 后端报错: ${data.content}`;
                eventSource.close();
            }
        } catch (e) {
            console.error("JSON解析失败:", event.data, e);
        }
    };

    eventSource.onerror = (error) => {
        console.error("EventSource 错误:", error);
        if (isFirstText) msgEl.textContent = `❌ 连接中断，请检查后端或代理软件`;
        eventSource.close();
    };

    const originalClose = eventSource.close.bind(eventSource);
    eventSource.close = () => {
        sendBtn.disabled = false;
        sendBtn.textContent = '发送';
        userInput.focus();
        originalClose();
    };
}

// ==================== 辅助渲染 ====================
function addMessage(role, content, isLoading = false, steps = null, scroll = true) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.textContent = content;
    if (isLoading) msg.innerHTML = `<span class="loading">${content}</span>`;
    if (steps && steps.length > 0) renderSteps(msg, steps);
    chatMessages.appendChild(msg);
    if (scroll) chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function renderSteps(container, steps) {
    const stepsEl = document.createElement('details');
    stepsEl.className = 'steps';
    const summary = document.createElement('summary');
    summary.textContent = `🔧 执行过程 (${steps.length} 步)`;
    stepsEl.appendChild(summary);
    
    steps.forEach(step => {
        const item = document.createElement('div');
        item.className = 'step-item';
        if (step.action === 'tool') {
            item.innerHTML = `<strong>Step ${step.step}:</strong> 调用 <code>${step.tool}(${step.args})</code>`;
            if (step.tool === 'generate_image' && step.result && step.result.image_url) {
                const img = document.createElement('img');
                img.src = step.result.image_url;
                img.style.maxWidth = '100%';
                img.style.marginTop = '8px';
                img.style.borderRadius = '8px';
                item.appendChild(img);
            }
        } else if (step.action === 'final') {
            item.innerHTML = `<strong>Step ${step.step}:</strong> 生成最终回答`;
        }
        stepsEl.appendChild(item);
    });
    container.appendChild(stepsEl);
}

async function loadKnowledge() {
    loadKnowledgeBtn.disabled = true;
    loadKnowledgeBtn.textContent = '加载中...';
    try {
        const response = await fetch(`${API_BASE}/knowledge/load`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: 'knowledge.txt' })
        });
        const data = await response.json();
        if (response.ok) {
            await updateStats();
            addMessage('assistant', '✅ 知识库加载成功！');
        } else {
            addMessage('assistant', `❌ 加载失败: ${data.detail}`);
        }
    } catch (error) { addMessage('assistant', `❌ 网络错误: ${error.message}`); } 
    finally {
        loadKnowledgeBtn.disabled = false;
        loadKnowledgeBtn.textContent = '加载 knowledge.txt';
    }
}

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE}/knowledge/stats`);
        const data = await response.json();
        knowledgeStats.textContent = `📄 ${data['文档数量']} 个文档`;
    } catch (error) { knowledgeStats.textContent = '统计失败'; }
}

if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        uploadBtn.disabled = true;
        uploadBtn.textContent = '⏳ 上传中...';
        addMessage('assistant', `📎 正在上传并解析 ${file.name}...`);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch(`${API_BASE}/knowledge/upload`, { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                await updateStats();
                addMessage('assistant', `✅ 已上传 ${file.name}\n📄 当前共 ${data.stats['文档数量']} 个文档块`);
            } else {
                addMessage('assistant', `❌ 上传失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) { addMessage('assistant', `❌ 网络错误: ${error.message}`); } 
        finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '📎 上传文档';
            fileInput.value = ''; 
        }
    });
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
loadKnowledgeBtn.addEventListener('click', loadKnowledge);
newChatBtn.addEventListener('click', newChat);

document.querySelectorAll('.example').forEach(btn => {
    btn.addEventListener('click', () => { userInput.value = btn.dataset.q; userInput.focus(); });
});

window.addEventListener('load', async () => { await updateStats(); await loadSessionList(); });