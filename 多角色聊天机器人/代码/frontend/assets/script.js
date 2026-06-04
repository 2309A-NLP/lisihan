// API基础URL
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// 全局变量
let currentUserId = 1; // 默认用户ID
let currentRoleId = 1; // 默认角色ID
let conversationId = null;

// 角色信息映射
const roleInfo = {
    1: { name: '医生', icon: 'fa-stethoscope', color: 'bg-blue-500' },
    2: { name: '律师', icon: 'fa-balance-scale', color: 'bg-purple-500' },
    3: { name: '心理医生', icon: 'fa-heart', color: 'bg-green-500' },
    4: { name: '教师', icon: 'fa-graduation-cap', color: 'bg-yellow-500' },
    5: { name: '科学家', icon: 'fa-flask', color: 'bg-indigo-500' },
    6: { name: '股票分析师', icon: 'fa-line-chart', color: 'bg-red-500' },
    7: { name: '英语学习助手', icon: 'fa-language', color: 'bg-pink-500' },
    8: { name: '虚拟朋友', icon: 'fa-smile-o', color: 'bg-orange-500' },
    9: { name: '金融理财师', icon: 'fa-money', color: 'bg-teal-500' }
};

// DOM元素
const chatBody = document.getElementById('chat-body');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const roleItems = document.querySelectorAll('.role-item');
const currentRoleAvatar = document.getElementById('current-role-avatar');
const currentRoleName = document.getElementById('current-role-name');

// 初始化
// Function: Initialize chat page events, default role state, and history loading.
function init() {
    // 绑定事件
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 绑定角色选择事件
    roleItems.forEach(item => {
        item.addEventListener('click', () => {
            const roleId = parseInt(item.dataset.roleId);
            handleRoleChange(roleId);
            
            // 更新选中状态
            roleItems.forEach(i => i.classList.remove('bg-gray-100', 'border-l-4', 'border-primary'));
            item.classList.add('bg-gray-100', 'border-l-4', 'border-primary');
        });
    });
    
    // 默认选中第一个角色
    if (roleItems.length > 0) {
        roleItems[0].classList.add('bg-gray-100', 'border-l-4', 'border-primary');
    }
    
    // 加载初始历史记录
    loadHistory();
}

// 处理角色切换
// Function: Switch the active role, reset the conversation id, and refresh role UI state.
function handleRoleChange(roleId) {
    currentRoleId = roleId;
    conversationId = null;
    
    // 更新当前角色信息
    const role = roleInfo[roleId];
    if (role) {
        currentRoleName.textContent = role.name;
        currentRoleAvatar.className = `role-avatar ${role.color}`;
        currentRoleAvatar.innerHTML = `<i class="fa ${role.icon}"></i>`;
    }
    
    // 清空聊天记录
    chatBody.innerHTML = '<div class="message system mb-4 flex justify-center"><div class="bg-neutral-dark text-gray-600 px-4 py-2 rounded-full text-sm max-w-xs">角色已切换，开始新的对话</div></div>';
    
    // 加载新角色的历史记录
    loadHistory();
}

// 加载历史记录
// Function: Load recent history for the current user and role from the backend.
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/history`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUserId,
                role_id: currentRoleId
            })
        });
        
        if (!response.ok) {
            throw new Error('加载历史记录失败');
        }
        
        const data = await response.json();
        
        // 清空聊天记录
        chatBody.innerHTML = '';
        
        // 添加历史消息
        if (data.history && data.history.length > 0) {
            data.history.forEach(msg => {
                addMessage(msg.sender, msg.content, msg.timestamp);
            });
        } else {
            // 添加欢迎消息
            addMessage('system', '欢迎使用多角色聊天机器人，开始对话吧！');
        }
        
        // 滚动到底部
        scrollToBottom();
    } catch (error) {
        console.error('加载历史记录出错:', error);
        addMessage('system', '加载历史记录失败，请刷新页面重试');
    }
}

// 发送消息
// Function: Send user input to the chat API and render the returned assistant reply.
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // 禁用发送按钮，防止重复发送
    sendButton.disabled = true;
    sendButton.textContent = '发送中...';
    
    // 添加用户消息到界面
    const timestamp = Math.floor(Date.now() / 1000);
    const userMessageElement = addMessage('user', message, timestamp);
    
    // 添加消息状态
    const statusElement = document.createElement('div');
    statusElement.className = 'message-status';
    statusElement.textContent = '发送中...';
    userMessageElement.appendChild(statusElement);
    
    // 清空输入框
    messageInput.value = '';
    
    // 先创建一个空回复气泡，后续 SSE token 会持续追加进去。
    const replyElement = addMessage('role', '', Math.floor(Date.now() / 1000));
    const replyBubble = replyElement.querySelector('.message-bubble');
    if (replyBubble) {
        replyBubble.textContent = '正在思考...';
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: currentUserId,
                role_id: currentRoleId,
                message: message,
                conversation_id: conversationId
            })
        });
        
        if (!response.ok) {
            throw new Error('发送消息失败');
        }

        const result = await readChatStream(response, (chunk) => {
            if (!replyBubble) return;
            if (replyBubble.textContent === '正在思考...') {
                replyBubble.textContent = '';
            }
            replyBubble.textContent += chunk;
            scrollToBottom();
        });
        conversationId = result.conversation_id || conversationId;
        
        // 更新消息状态为已发送
        statusElement.className = 'message-status sent';
        statusElement.textContent = '已发送';

        if (replyBubble && !replyBubble.textContent.trim()) {
            replyBubble.textContent = result.response || '我这边暂时没有生成回复，可以再试一次。';
        }
        
        // 滚动到底部
        scrollToBottom();
    } catch (error) {
        console.error('发送消息出错:', error);
        
        // 更新消息状态为错误
        statusElement.className = 'message-status error';
        statusElement.textContent = '发送失败';
        
        if (replyElement) {
            replyElement.remove();
        }
        
        // 显示错误消息
        addMessage('system', '发送消息失败，请检查网络连接后重试');
        
        // 滚动到底部
        scrollToBottom();
    } finally {
        // 恢复发送按钮状态
        sendButton.disabled = false;
        sendButton.textContent = '发送';
    }
}

// 读取后端 SSE 流，边收到 token 边渲染。
async function readChatStream(response, onToken) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    const result = { response: '', conversation_id: null };

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
            const parsed = parseSSE(rawEvent);
            if (!parsed) continue;
            if (parsed.event === 'token') {
                const chunk = parsed.data.content || '';
                result.response += chunk;
                onToken(chunk);
            } else if (parsed.event === 'meta') {
                result.conversation_id = parsed.data.conversation_id || result.conversation_id;
            } else if (parsed.event === 'done') {
                result.conversation_id = parsed.data.conversation_id || result.conversation_id;
                result.response = parsed.data.response || result.response;
            } else if (parsed.event === 'error') {
                throw new Error(parsed.data.detail || '发送消息失败');
            }
        }
    }
    return result;
}

function parseSSE(rawEvent) {
    const lines = rawEvent.split('\n');
    let event = 'message';
    let data = '';
    for (const line of lines) {
        if (line.startsWith('event:')) {
            event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            data += line.slice(5).trim();
        }
    }
    if (!data) return null;
    return { event, data: JSON.parse(data) };
}

// 添加消息到界面
// Function: Render one message bubble according to sender type.
function addMessage(sender, content, timestamp = null) {
    const messageDiv = document.createElement('div');
    
    if (sender === 'system') {
        messageDiv.className = 'message system mb-4 flex justify-center';
        messageDiv.innerHTML = `
            <div class="bg-neutral-dark text-gray-600 px-4 py-2 rounded-full text-sm max-w-xs">
                ${content}
            </div>
        `;
    } else if (sender === 'user') {
        messageDiv.className = 'message user mb-4 flex justify-end items-end gap-2';
        messageDiv.innerHTML = `
            <div class="flex flex-col items-end max-w-[70%]">
                <div class="message-bubble bg-blue-100 text-gray-800 px-4 py-2 rounded-lg rounded-br-none message-shadow whitespace-pre-line">
                    ${content}
                </div>
                ${timestamp ? `<div class="text-xs text-gray-500 mt-1">${formatTimestamp(timestamp)}</div>` : ''}
            </div>
            <div class="user-avatar">
                <i class="fa fa-user"></i>
            </div>
        `;
    } else if (sender === 'role') {
        messageDiv.className = 'message role mb-4 flex items-start gap-2';
        const role = roleInfo[currentRoleId];
        const roleColor = role ? role.color : 'bg-primary';
        const roleIcon = role ? role.icon : 'fa-stethoscope';
        
        messageDiv.innerHTML = `
            <div class="role-avatar ${roleColor}">
                <i class="fa ${roleIcon}"></i>
            </div>
            <div class="flex flex-col items-start max-w-[70%]">
                <div class="message-bubble bg-white text-gray-800 px-4 py-2 rounded-lg rounded-bl-none message-shadow border border-gray-200 whitespace-pre-line">
                    ${content}
                </div>
                ${timestamp ? `<div class="text-xs text-gray-500 mt-1">${formatTimestamp(timestamp)}</div>` : ''}
            </div>
        `;
    }
    
    chatBody.appendChild(messageDiv);
    
    return messageDiv;
}

// 添加加载消息
// Function: Render a temporary thinking indicator while waiting for the backend.
function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message role mb-4 flex items-start gap-2';
    loadingDiv.id = 'loading-message';
    
    const role = roleInfo[currentRoleId];
    const roleColor = role ? role.color : 'bg-primary';
    const roleIcon = role ? role.icon : 'fa-stethoscope';
    
    loadingDiv.innerHTML = `
        <div class="role-avatar ${roleColor}">
            <i class="fa ${roleIcon}"></i>
        </div>
        <div class="flex flex-col items-start max-w-[70%]">
            <div class="bg-white text-gray-800 px-4 py-2 rounded-lg rounded-bl-none message-shadow border border-gray-200">
                <div class="flex items-center gap-2">
                    <span>正在思考...</span>
                    <div class="w-4 h-4 border-2 border-gray-300 border-t-primary rounded-full animate-spin"></div>
                </div>
            </div>
        </div>
    `;
    
    chatBody.appendChild(loadingDiv);
    
    scrollToBottom();
    
    return loadingDiv;
}

// 滚动到底部
// Function: Scroll the chat container so the newest message is visible.
function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
}

// 格式化时间戳
// Function: Format a Unix timestamp as local hour and minute text.
function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', init);
