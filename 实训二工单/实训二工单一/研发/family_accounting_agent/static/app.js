const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const monthTotalEl = document.querySelector("#monthTotal");
const monthIncomeEl = document.querySelector("#monthIncome");
const todayTotalEl = document.querySelector("#todayTotal");
const personBarsEl = document.querySelector("#personBars");
const recentListEl = document.querySelector("#recentList");
const helpBtn = document.querySelector("#helpBtn");

const welcome =
  "您好，欢迎使用咱们小家专属记账本！请按照“x年x月x日，谁做什么事收入/支出多少钱”的格式来输入。请告诉我你的账目需求吧。";

function money(value) {
  const number = Number(value || 0);
  return `¥${Number.isInteger(number) ? number : number.toFixed(2)}`;
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  messagesEl.appendChild(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  input.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    addMessage("bot", data.reply || "我没有收到有效回复。");
    if (data.summary) renderSummary(data.summary);
  } catch (error) {
    addMessage("bot", "服务暂时不可用，请确认后端和 MySQL 正在运行。");
  } finally {
    input.disabled = false;
    input.focus();
  }
}

function renderSummary(summary) {
  monthTotalEl.textContent = money(summary.month_total);
  monthIncomeEl.textContent = money(summary.month_income);
  todayTotalEl.textContent = money(summary.today_total);

  const maxPerson = Math.max(...(summary.by_member || []).map((item) => item.total), 1);
  personBarsEl.innerHTML = "";
  if (!summary.by_member || summary.by_member.length === 0) {
    personBarsEl.innerHTML = '<p class="empty">本月还没有支出记录。</p>';
  } else {
    summary.by_member.forEach((item) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-meta"><span>${item.member}</span><strong>${money(item.total)}</strong></div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(8, (item.total / maxPerson) * 100)}%"></div></div>
      `;
      personBarsEl.appendChild(row);
    });
  }

  recentListEl.innerHTML = "";
  if (!summary.recent || summary.recent.length === 0) {
    recentListEl.innerHTML = '<p class="empty">暂无明细，先记一笔吧。</p>';
  } else {
    summary.recent.forEach((item) => {
      const row = document.createElement("div");
      row.className = `recent-item ${item.type === "收入" ? "income" : "expense"}`;
      row.innerHTML = `<strong>#${item.id} ${item.item} ${money(item.amount)}</strong>${item.date} · ${item.member} · ${item.type}`;
      recentListEl.appendChild(row);
    });
  }
}

async function loadSummary() {
  try {
    const response = await fetch("/api/summary");
    renderSummary(await response.json());
  } catch (error) {
    personBarsEl.innerHTML = '<p class="empty">无法读取概览，请确认后端和 MySQL 已启动。</p>';
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

document.querySelectorAll("[data-message]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.message));
});

helpBtn.addEventListener("click", () => {
  sendMessage("帮助");
});

addMessage("bot", welcome);
loadSummary();
