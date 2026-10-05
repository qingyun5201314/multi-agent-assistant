const API_BASE = '';

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loadKnowledgeBtn = document.getElementById('loadKnowledge');
const knowledgeStats = document.getElementById('knowledgeStats');
const clearChatBtn = document.getElementById('clearChat');
const maxStepsInput = document.getElementById('maxSteps');

// ==================== 发送消息 ====================
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;
    
    userInput.value = '';
    sendBtn.disabled = true;
    sendBtn.textContent = '思考中';
    
    addMessage('user', query);
    const loadingEl = addMessage('assistant', '🤔 思考中...', true);
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_steps: parseInt(maxStepsInput.value) || 5
            })
        });
        
        const data = await response.json();
        loadingEl.remove();
        
        if (data.success) {
            addMessage('assistant', data.answer, false, data.steps);
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
function addMessage(role, content, isLoading = false, steps = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    if (isLoading) {
        msg.innerHTML = `<span class="loading">${content}</span>`;
    } else {
        msg.textContent = content;
    }
    
    // 处理工具调用步骤
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
                
                // ⭐ 关键：如果生成了图片，显示出来
                if (step.tool === 'generate_image' && step.result && step.result.image_url) {
                    const imgContainer = document.createElement('div');
                    imgContainer.style.marginTop = '10px';
                    
                    const loadingText = document.createElement('div');
                    loadingText.textContent = '🎨 图像生成中，请稍候（5-15秒）...';
                    loadingText.style.fontSize = '12px';
                    loadingText.style.color = '#78350f';
                    
                    const img = document.createElement('img');
                    img.src = step.result.image_url;
                    img.style.maxWidth = '100%';
                    img.style.marginTop = '8px';
                    img.style.borderRadius = '8px';
                    img.style.display = 'none';
                    
                    // 图片加载完成后显示
                    img.onload = () => {
                        loadingText.remove();
                        img.style.display = 'block';
                    };
                    
                    // 图片加载失败
                    img.onerror = () => {
                        loadingText.textContent = '❌ 图像加载失败，请重试';
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
    chatMessages.scrollTop = chatMessages.scrollHeight;
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
            addMessage('assistant', '✅ 知识库加载成功！现在可以问关于知识库的问题了。');
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

// ==================== 清空对话 ====================
function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome">
            <h2>👋 你好！我是多功能AI助手</h2>
            <p>对话已清空，继续提问吧！</p>
        </div>
    `;
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
clearChatBtn.addEventListener('click', clearChat);

document.querySelectorAll('.example').forEach(btn => {
    btn.addEventListener('click', () => {
        userInput.value = btn.dataset.q;
        userInput.focus();
    });
});

window.addEventListener('load', updateStats);