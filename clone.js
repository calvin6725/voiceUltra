(() => {
  const get = id => document.getElementById(id);
  const clone = { source: null, reference: null, profiles: [], ready: false, busy: false, url: null, timer: null };
  const stages = {
    preparing_reference: '正在整理参考片段', loading_recognizer: '正在载入原话识别模型',
    transcribing: '正在识别参考原话', loading_model: '正在载入克隆模型',
    synthesizing: '正在生成克隆配音'
  };
  const qualityNames = { fast: '快速试听', standard: '音色优先', full: '实验模式' };
  const message = (id, text, error = false) => {
    get(id).textContent = text;
    get(id).classList.toggle('error', error);
  };
  function update() {
    get('prepareClone').disabled = !clone.source || !clone.ready || clone.busy;
    get('saveClone').disabled = !clone.reference || !get('cloneName').value.trim() || !get('cloneTranscript').value.trim() || clone.busy;
    const changed = clone.reference && (!clone.reference.saved ||
      clone.reference.text !== get('cloneTranscript').value.trim() ||
      clone.reference.name !== get('cloneName').value.trim());
    get('generateClone').disabled = !clone.ready || !clone.reference?.saved || changed || !get('cloneText').value.trim() || clone.busy;
    get('cloneCount').textContent = `${get('cloneText').value.length} / 3000`;
    get('cloneSpeedValue').textContent = `${Number(get('cloneSpeed').value).toFixed(2)} 倍`;
    get('generateCloneLabel').textContent = clone.busy ? '正在处理…' : '生成克隆配音';
    for (const id of ['cloneStart', 'cloneLength', 'cloneName', 'cloneTranscript', 'cloneText', 'cloneQuality', 'cloneSteps', 'cloneSpeed']) get(id).disabled = clone.busy;
    get('cloneProfiles').querySelectorAll('button').forEach(button => button.disabled = clone.busy);
  }
  async function api(path, options) {
    const response = await fetch(path, options);
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error || '本地服务未能完成请求');
    }
    return response;
  }
  function json(payload) {
    return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
  }
  function clearResult() {
    get('cloneResultAudio').pause();
    get('cloneResultAudio').removeAttribute('src');
    get('cloneResultAudio').load();
    if (clone.url) URL.revokeObjectURL(clone.url);
    clone.url = null;
    get('cloneResult').hidden = true;
  }
  function select(reference) {
    clone.reference = reference;
    get('cloneTranscript').value = reference.text;
    get('cloneName').value = (reference.saved ? reference.name : reference.name.replace(/\.[^.]+$/, '')).slice(0, 50);
    get('cloneReferenceAudio').src = `/api/clone/reference/${reference.id}`;
    get('cloneReferenceEditor').hidden = false;
    message('cloneWarning', (reference.warnings || []).join('；'));
    message('cloneStatus', `参考片段 ${reference.duration.toFixed(1)} 秒 · ${reference.saved ? '音色已保存' : '请核对原话后保存音色'}`);
    clearResult();
    renderProfiles();
    update();
  }
  function renderProfiles() {
    get('cloneProfiles').replaceChildren(...clone.profiles.map(profile => {
      const row = document.createElement('div');
      row.className = 'clone-profile-row';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'clone-profile';
      button.textContent = profile.name;
      button.setAttribute('aria-pressed', String(profile.id === clone.reference?.id));
      button.addEventListener('click', () => {
        select(profile);
        try { localStorage.setItem('voiceLabCloneActive', profile.id); } catch {}
      });
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'icon-btn';
      remove.title = `删除克隆音色：${profile.name}`;
      remove.setAttribute('aria-label', remove.title);
      const icon = document.createElement('i');
      icon.dataset.lucide = 'trash-2';
      remove.append(icon);
      remove.addEventListener('click', () => operation(async () => {
        await api(`/api/clone/profile/${profile.id}`, { method: 'DELETE' });
        clone.profiles = clone.profiles.filter(item => item.id !== profile.id);
        if (clone.reference?.id === profile.id) {
          clone.reference = null;
          clearResult();
          get('cloneReferenceEditor').hidden = true;
        }
        renderProfiles();
        message('cloneProgress', '克隆音色已删除');
      }));
      row.append(button, remove);
      return row;
    }));
    refreshIcons();
  }
  async function connect() {
    try {
      const data = await (await api('/api/clone/status', { cache: 'no-store' })).json();
      clone.ready = data.qualities.length > 0;
      clone.profiles = data.profiles;
      for (const option of get('cloneQuality').options) option.disabled = !data.qualities.includes(option.value);
      const preferred = data.qualities.includes('standard') ? 'standard' : data.qualities[0];
      if (preferred) get('cloneQuality').value = preferred;
      get('cloneEngine').textContent = clone.ready ? '本地 AI 克隆' : '模型尚未安装';
      let active;
      try { active = localStorage.getItem('voiceLabCloneActive'); } catch {}
      const profile = clone.profiles.find(item => item.id === active) || clone.profiles[0];
      if (profile) select(profile);
      else renderProfiles();
      if (!clone.ready) message('cloneStatus', '请先安装本地声音克隆模型', true);
    } catch {
      clone.ready = false;
      get('cloneEngine').textContent = '本地服务未连接';
      message('cloneStatus', '请通过“启动声线实验室.bat”打开新版声音克隆服务', true);
    }
    update();
  }
  async function operation(work) {
    clone.busy = true;
    update();
    const started = Date.now();
    const poll = async () => {
      try {
        const data = await (await api('/api/clone/status')).json();
        if (!clone.busy) return;
        const stage = stages[data.activity.stage] || '正在处理';
        const progress = data.activity.progress > 0 ? ` · ${Math.round(data.activity.progress * 100)}%` : '';
        message('cloneProgress', `${stage}${progress} · 已用时 ${Math.floor((Date.now() - started) / 1000)} 秒`);
      } catch {}
    };
    clone.timer = setInterval(poll, 1500);
    try { await work(); }
    catch (error) { message('cloneProgress', error.message || '处理失败，请重试', true); }
    finally { clearInterval(clone.timer); clone.busy = false; update(); }
  }
  window.addEventListener('voice-reference-loaded', event => {
    clone.source = event.detail;
    get('cloneStart').value = '0';
    get('cloneLength').value = String(Math.min(12, event.detail.duration).toFixed(1));
    get('cloneStart').max = String(Math.max(0, event.detail.duration - 3));
    message('cloneStatus', `已载入 ${event.detail.name} · 请选择参考片段`);
    update();
  });
  window.addEventListener('voice-reference-loading', () => {
    clone.source = null;
    update();
  });
  get('prepareClone').addEventListener('click', () => operation(async () => {
    const query = new URLSearchParams({
      name: clone.source.name, start: get('cloneStart').value, duration: get('cloneLength').value
    });
    message('cloneProgress', '正在提取参考片段并识别原话…');
    const reference = await (await api(`/api/clone/reference?${query}`, { method: 'POST', body: clone.source.blob })).json();
    select(reference);
    message('cloneProgress', reference.text ? '原话识别完成，请试听并核对后保存' : '请填写参考片段实际说出的原话');
  }));
  get('saveClone').addEventListener('click', () => operation(async () => {
    const reference = await (await api('/api/clone/profile', json({
      id: clone.reference.id, name: get('cloneName').value, text: get('cloneTranscript').value
    }))).json();
    clone.profiles = [reference, ...clone.profiles.filter(item => item.id !== reference.id)];
    select(reference);
    try { localStorage.setItem('voiceLabCloneActive', reference.id); } catch {}
    message('cloneProgress', '克隆音色已保存');
  }));
  get('generateClone').addEventListener('click', () => operation(async () => {
    clearResult();
    message('cloneProgress', '正在准备克隆配音…');
    const quality = get('cloneQuality').value;
    const response = await api('/api/clone/generate', json({
      id: clone.reference.id, text: get('cloneText').value,
      quality, speed: Number(get('cloneSpeed').value), steps: Number(get('cloneSteps').value)
    }));
    const blob = await response.blob();
    clone.url = URL.createObjectURL(blob);
    get('cloneResultAudio').src = clone.url;
    get('cloneResultName').textContent = `${clone.reference.name} · 克隆配音`;
    get('cloneTiming').textContent = `${qualityNames[quality]} · 生成 ${response.headers.get('X-Elapsed-Seconds')} 秒 · 音频 ${response.headers.get('X-Audio-Seconds')} 秒`;
    get('cloneResult').hidden = false;
    message('cloneProgress', '配音已生成');
    try {
      const request = indexedDB.open('voiceLabAudio', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('clips');
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction('clips', 'readwrite');
        transaction.objectStore('clips').put({ blob, name: `${clone.reference.name}-克隆配音.wav` }, 'latest');
        transaction.oncomplete = () => database.close();
      };
    } catch {}
  }));
  get('downloadClone').addEventListener('click', () => {
    if (!clone.url) return;
    const link = document.createElement('a');
    link.href = clone.url;
    link.download = `${clone.reference.name.replace(/[<>:"/\\|?*]/g, '-')}-克隆配音.wav`;
    link.click();
  });
  for (const id of ['cloneTranscript', 'cloneName', 'cloneText', 'cloneSpeed', 'cloneQuality', 'cloneSteps']) {
    get(id).addEventListener('input', () => {
      update();
      if (['cloneText', 'cloneSpeed', 'cloneQuality', 'cloneSteps'].includes(id)) clearResult();
      if (id === 'cloneText') { try { localStorage.setItem('voiceLabCloneDraft', get(id).value); } catch {} }
    });
  }
  get('cloneQuality').addEventListener('change', () => {
    if (get('cloneQuality').value === 'full') get('cloneSteps').value = '16';
    update();
  });
  get('cloneReferenceAudio').addEventListener('play', () => { get('cloneResultAudio').pause(); stop(true); });
  get('cloneResultAudio').addEventListener('play', () => { get('cloneReferenceAudio').pause(); stop(true); });
  try { get('cloneText').value = localStorage.getItem('voiceLabCloneDraft') || ''; } catch {}
  window.addEventListener('pagehide', () => { clearInterval(clone.timer); if (clone.url) URL.revokeObjectURL(clone.url); });
  update();
  connect();
})();
