/* 风电混塔表面展开工具 - 前端逻辑 */
const $ = (id) => document.getElementById(id);
let previewMeta = null;   // {W,H}
let doorX = null, doorY = null;
let zMin = 0, zMax = 0;

// ---------- 进度轮询 ----------
let pollTimer = null;
function watchJob(jid, onDone) {
  $("progressBar").hidden = false;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/job/${jid}`).then(r => r.json());
      if (r.error && !r.kind) { clearInterval(pollTimer); $("progressBar").hidden = true; alert(r.error); return; }
      $("progressStage").textContent = r.stage;
      $("progressFill").style.width = `${Math.round(r.pct * 100)}%`;
      if (r.done) {
        clearInterval(pollTimer);
        $("progressBar").hidden = true;
        if (r.error) alert(`失败：${r.error}`);
        else onDone(r.result);
      }
    } catch (e) { /* 网络抖动忽略 */ }
  }, 800);
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

// ---------- 1. 加载模型 ----------
$("btnLoad").onclick = async () => {
  const path = $("modelPath").value.trim();
  if (!path) return alert("请先输入模型路径");
  $("btnLoad").disabled = true;
  const r = await post("/api/load", { path });
  if (r.error) { $("btnLoad").disabled = false; return alert(r.error); }
  watchJob(r.job_id, (res) => {
    $("btnLoad").disabled = false;
    zMin = res.z_min; zMax = res.z_max;
    const zs = res.z_suggest || [res.z_min, res.z_max];
    $("loadInfo").innerHTML =
      `✅ 已加载 <b>${res.n_blocks}</b> 个分块｜塔高 z = <b>${res.z_min.toFixed(1)} ~ ${res.z_max.toFixed(1)}</b> 米` +
      `｜半径 ${res.r_min.toFixed(2)} ~ ${res.r_max.toFixed(2)} 米` +
      `<br>📐 已自动选中混凝土主塔段 <b>${zs[0].toFixed(1)} ~ ${zs[1].toFixed(1)}</b> 米（可在下方修改，含钢塔段可手动调大）`;
    $("cardPreview").hidden = false;
    $("cardRender").hidden = false;
    $("zBottom").value = zs[0].toFixed(1);
    $("zTop").value = zs[1].toFixed(1);
    updateEstimate();
  });
};

// ---------- 2. 预览 + 门洞点选 ----------
$("btnPreview").onclick = async () => {
  $("btnPreview").disabled = true;
  const r = await post("/api/preview", { px_m: 0.02 });
  if (r.error) { $("btnPreview").disabled = false; return alert(r.error); }
  watchJob(r.job_id, (res) => {
    $("btnPreview").disabled = false;
    previewMeta = res;
    const img = $("previewImg");
    img.src = `${res.url}?t=${Date.now()}`;
    img.onload = () => {
      // 限制显示宽度，点击坐标按比例换算回像素
      const maxW = $("previewScroll").clientWidth - 4;
      const scale = Math.min(1, maxW / res.W, 560 / res.H * (res.W / res.W));
      img.style.width = `${Math.round(res.W * Math.min(1, maxW / res.W))}px`;
      $("previewWrap").hidden = false;
    };
  });
};

$("previewImg").addEventListener("click", (e) => {
  if (!previewMeta) return;
  const img = e.target;
  const rect = img.getBoundingClientRect();
  const scaleX = previewMeta.W / rect.width;
  const scaleY = previewMeta.H / rect.height;
  doorX = Math.round((e.clientX - rect.left) * scaleX);
  doorY = Math.round((e.clientY - rect.top) * scaleY);
  const line = $("doorLine");
  line.hidden = false;
  line.style.left = `${e.clientX - rect.left}px`;
  $("doorInfo").innerHTML = `📍 门洞已标记（像素 x=${doorX}）。正式展开图将以门洞为横向中心。如需调整，再次点击即可。`;
});

// ---------- 3. 参数与渲染 ----------
function updateEstimate() {
  const z0 = parseFloat($("zBottom").value), z1 = parseFloat($("zTop").value);
  const pxm = parseFloat($("pxm").value), ss = parseInt($("ss").value);
  if (isNaN(z0) || isNaN(z1) || z1 <= z0) { $("estimate").textContent = ""; return; }
  const rows = (z1 - z0) / pxm;
  // 经验值：2mm/px ss=2 约 27ms/行（全周约1万像素宽）
  const secs = rows * 0.027 * ss * ss * (pxm / 0.002);
  const mp = (rows * 2 * Math.PI * 4 / pxm) / 1e6;
  const split = parseFloat($("split").value);
  let warn = "";
  if (split === 0 && mp > 600)
    warn = `<br><span class="accent">⚠️ 单张约 ${(mp/1000).toFixed(1)} 亿像素，文件非常大（PS可能打不开），建议选择分张导出</span>`;
  $("estimate").innerHTML =
    `预计输出约 ${(rows / 1000).toFixed(1)}k 行、${mp.toFixed(0)} 百万像素/张；` +
    `参考耗时约 ${secs < 90 ? Math.round(secs) + " 秒" : (secs / 60).toFixed(0) + " 分钟"}/10米塔段（实际取决于塔径）` + warn;
}
["zBottom", "zTop", "pxm", "ss", "split"].forEach(id => $(id).addEventListener("input", updateEstimate));
$("strength").addEventListener("input", () => $("strengthVal").textContent = $("strength").value + "%");
$("clarity").addEventListener("input", () => $("clarityVal").textContent = $("clarity").value + "%");
$("engine").addEventListener("change", async () => {
  if ($("engine").value !== "blender") return;
  const r = await fetch("/api/blender_status").then(r => r.json());
  if (!r.found) alert("未检测到本机安装 Blender 4.x。\n烘焙模式需要 Blender，可到 blender.org 下载安装，\n或改用内置快速渲染。");
});

$("btnRender").onclick = async () => {
  const body = {
    z_bottom: parseFloat($("zBottom").value),
    z_top: parseFloat($("zTop").value),
    px_m: parseFloat($("pxm").value),
    ss: parseInt($("ss").value),
    split_m: parseFloat($("split").value),
    name: $("outName").value.trim() || "tower",
    strength: parseInt($("strength").value) / 100,
    clarity: parseInt($("clarity").value) / 100,
    engine: $("engine").value,
    door_x: doorX, door_y: doorY,
  };
  if (isNaN(body.z_bottom) || isNaN(body.z_top) || body.z_top <= body.z_bottom)
    return alert("请检查高度范围");
  if (doorX === null && !confirm("还没有点选门洞，展开图将不居中门洞。继续？")) return;
  $("btnRender").disabled = true;
  const r = await post("/api/render", body);
  if (r.error) { $("btnRender").disabled = false; return alert(r.error); }
  watchJob(r.job_id, (res) => {
    $("btnRender").disabled = false;
    $("cardResult").hidden = false;
    $("resultList").innerHTML = res.files.map(f =>
      `<div class="result-item">
         <span>🖼️</span>
         <a href="${f.url}" download>${f.name}</a>
         <span class="dim">${f.W} × ${f.H} 像素</span>
       </div>`).join("") +
      `<div class="info">文件保存在软件目录 output\\app\\ 下，点文件名可直接下载。门洞方位角 θ₀=${res.theta0.toFixed(3)} rad</div>`;
    $("cardResult").scrollIntoView({ behavior: "smooth" });
  });
};
