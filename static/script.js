const API_BASE = '';

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadKnowledgeBtn = document.getElementById('loadKnowledge');
const knowledgeStats = document.getElementById('knowledgeStats');
const maxStepsInput = document.getElementById('maxSteps');
const sessionList = document.getElementById('sessionList');
const newChatBtn = document.getElementById('newChatBtn');

// ==================== 会话管理 ====================
let currentSessionId = null;

function generateSessionId() {
    return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// ==================== 加载会话列表 ====================
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
            if (session.session_id === currentSessionId) {
                item.classList.add('active');
            }
            
            item.innerHTML = `
                <div class="session-title">${session.title || '新对话'}</div>
                <button class="session-delete" title="删除">×</button>
            `;
            
            item.querySelector('.session-title').addEventListener('click', () => {
                switchSession(session.session_id);
            });
            
            item.querySelector('.session-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('确定删除这个对话吗？')) {
                    deleteSession(session.session_id);
                }
            });
            
            sessionList.appendChild(item);
        });
    } catch (error) {
        console.error('加载会话列表失败:', error);
    }
}

// ==================== 切换会话 ====================
async function switchSession(sessionId) {
    currentSessionId = sessionId;
    
    try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
        const data = await response.json();
        
        chatMessages.innerHTML = '';
        
        if (data.messages.length === 0) {
            chatMessages.innerHTML = `
                <div class="welcome">
                    <h2>👋 新对话</h2>
                    <p>继续提问吧！</p>
                </div>
            `;
        } else {
            data.messages.forEach(msg => {
                addMessage(msg.role, msg.content, false, null, false);
            });
            
            // 滚动到底部（简单版）
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        }
        
        loadSessionList();
    } catch (error) {
        console.error('加载会话消息失败:', error);
    }
}

// ==================== 新对话 ====================
async function newChat() {
    currentSessionId = generateSessionId();
    
    chatMessages.innerHTML = `
        <div class="welcome">
            <h2>👋 新对话</h2>
            <p>开始提问吧！</p>
        </div>
    `;
    
    await loadSessionList();
}

// ==================== 删除会话 ====================
async function deleteSession(sessionId) {
    try {
        await fetch(`${API_BASE}/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (sessionId === currentSessionId) {
            await newChat();
        } else {
            await loadSessionList();
        }
    } catch (error) {
        console.error('删除失败:', error);
    }
}

// ==================== 发送消息 ====================
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;
    
    if (!currentSessionId) {
        currentSessionId = generateSessionId();
    }
    
    userInput.value = '';
    sendBtn.disabled = true;
    sendBtn.textContent = '思考中';
    
    const welcome = chatMessages.querySelector('.welcome');
    if (welcome) welcome.remove();
    
    addMessage('user', query, false, null, true);
    const loadingEl = addMessage('assistant', '🤔 思考中...', true);
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId,
                max_steps: parseInt(maxStepsInput.value) || 5
            })
        });
        
        const data = await response.json();
        loadingEl.remove();
        
        if (data.success) {
            addMessage('assistant', data.answer, false, data.steps, true);
            await loadSessionList();
        } else {
            addMessage('assistant', `❌ 出错了: ${data.error || '未知错误'}`);
        }
    } catch (error) {
        loadingEl.remove();
        addMessage('assistant', `❌ 网络错误: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = '发送';
        userInput.focus();
    }
}

// ==================== 添加消息到界面 ====================
function addMessage(role, content, isLoading = false, steps = null, scroll = true) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    if (isLoading) {
        msg.innerHTML = `<span class="loading">${content}</span>`;
    } else {
        msg.textContent = content;
    }
    
    if (steps && steps.length > 0) {
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
                    const imgContainer = document.createElement('div');
                    imgContainer.style.marginTop = '10px';
                    
                    const loadingText = document.createElement('div');
                    loadingText.textContent = '🎨 图像生成中，请稍候...';
                    loadingText.style.fontSize = '12px';
                    loadingText.style.color = '#78350f';
                    
                    const img = document.createElement('img');
                    img.src = step.result.image_url;
                    img.style.maxWidth = '100%';
                    img.style.marginTop = '8px';
                    img.style.borderRadius = '8px';
                    img.style.display = 'none';
                    
                    img.onload = () => {
                        loadingText.remove();
                        img.style.display = 'block';
                    };
                    
                    img.onerror = () => {
                        loadingText.textContent = '❌ 图像加载失败';
                        loadingText.style.color = '#dc2626';
                    };
                    
                    imgContainer.appendChild(loadingText);
                    imgContainer.appendChild(img);
                    item.appendChild(imgContainer);
                }
            } else if (step.action === 'final') {
                item.innerHTML = `<strong>Step ${step.step}:</strong> 生成最终回答`;
            }
            
            stepsEl.appendChild(item);
        });
        
        msg.appendChild(stepsEl);
    }
    
    chatMessages.appendChild(msg);
    
    if (scroll) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    return msg;
}

// ==================== 加载知识库 ====================
async function loadKnowledge() {
    loadKnowledgeBtn.disabled = true;
    loadKnowledgeBtn.textContent = '加载中...';
    
    try {
        const response = await fetch(`${API_BASE}/knowledge/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: 'knowledge.txt' })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            await updateStats();
            addMessage('assistant', '✅ 知识库加载成功！');
        } else {
            addMessage('assistant', `❌ 加载失败: ${data.detail || '未知错误'}`);
        }
    } catch (error) {
        addMessage('assistant', `❌ 网络错误: ${error.message}`);
    } finally {
        loadKnowledgeBtn.disabled = false;
        loadKnowledgeBtn.textContent = '加载 knowledge.txt';
    }
}

// ==================== 更新知识库统计 ====================
async function updateStats() {
    try {
        const response = await fetch(`${API_BASE}/knowledge/stats`);
        const data = await response.json();
        knowledgeStats.textContent = `📄 ${data['文档数量']} 个文档`;
    } catch (error) {
        knowledgeStats.textContent = '统计失败';
    }
}

// ==================== 事件绑定 ====================
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

loadKnowledgeBtn.addEventListener('click', loadKnowledge);
newChatBtn.addEventListener('click', newChat);

// ==================== 页面加载 ====================
window.addEventListener('load', async () => {
    await updateStats();
    await loadSessionList();
});

// ==================== 文件上传 ====================
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');

uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // 显示上传中
    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ 上传中...';
    addMessage('assistant', `📎 正在上传并解析 ${file.name}...`);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/knowledge/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            await updateStats();
            addMessage('assistant', 
                `✅ 已上传 ${file.name}\n` +
                `📄 当前知识库共 ${data.stats['文档数量']} 个文档块\n` +
                `📚 来自 ${data.stats['文档源数量']} 个文件`
            );
        } else {
            addMessage('assistant', `❌ 上传失败: ${data.detail || '未知错误'}`);
        }
    } catch (error) {
        addMessage('assistant', `❌ 网络错误: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '📎 上传文档';
        fileInput.value = '';  // 清空，允许重复上传同一文件
    }
});