(function () {
  // ============= Constants (lock 配色) =============
  const COLOR_ORIGIN = '#2e7d32';
  const COLOR_DEST   = '#c62828';
  const COLOR_ROUTE  = '#1976d2';
  const COLOR_NGZ    = '#e57373';

  // ============= 地圖初始化 =============
  const map = L.map('map').setView([28, 135], 5);

  const tileOptions = {
    'cartodb-light': {
      url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      attr: '© CartoDB'
    },
    'esri-sat': {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attr: '© Esri'
    }
  };
  let currentTile = null;
  function setTile(name) {
    if (currentTile) map.removeLayer(currentTile);
    const t = tileOptions[name];
    currentTile = L.tileLayer(t.url, { maxZoom: 18, attribution: t.attr }).addTo(map);
  }
  setTile('cartodb-light');
  document.getElementById('tile-select').addEventListener('change', e => setTile(e.target.value));

  // ============= Markers / Layers state =============
  let mode = null;          // 'origin' | 'dest' | null
  let originMarker = null;
  let destMarker = null;
  const ngzLayers = L.featureGroup().addTo(map);
  let routeLayer = null;

  function mkIcon(color, label) {
    return L.divIcon({
      html: `<div style="background:${color};width:26px;height:26px;border-radius:50%;border:3px solid white;box-shadow:0 0 6px rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:12px">${label||''}</div>`,
      className: '',
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });
  }
  function setOrigin(latlng) {
    if (originMarker) map.removeLayer(originMarker);
    originMarker = L.marker(latlng, { icon: mkIcon(COLOR_ORIGIN, 'O') }).addTo(map);
    syncCoordInputs();
  }
  function setDest(latlng) {
    if (destMarker) map.removeLayer(destMarker);
    destMarker = L.marker(latlng, { icon: mkIcon(COLOR_DEST, 'D') }).addTo(map);
    syncCoordInputs();
  }
  function syncCoordInputs() {
    if (originMarker) {
      const ll = originMarker.getLatLng();
      document.getElementById('in-origin-lon').value = canonLon(ll.lng).toFixed(4);
      document.getElementById('in-origin-lat').value = ll.lat.toFixed(4);
    }
    if (destMarker) {
      const ll = destMarker.getLatLng();
      document.getElementById('in-dest-lon').value = canonLon(ll.lng).toFixed(4);
      document.getElementById('in-dest-lat').value = ll.lat.toFixed(4);
    }
  }

  // ============= Toolbar 按鈕 =============
  const btnOrigin = document.getElementById('btn-origin');
  const btnDest = document.getElementById('btn-dest');
  const btnNgz = document.getElementById('btn-ngz');
  const btnClear = document.getElementById('btn-clear');
  const btnRun = document.getElementById('btn-run');
  const btnApply = document.getElementById('btn-apply-coords');

  function setMode(m) {
    mode = m;
    [btnOrigin, btnDest].forEach(b => b.classList.remove('active'));
    if (m === 'origin') btnOrigin.classList.add('active');
    else if (m === 'dest') btnDest.classList.add('active');
  }
  btnOrigin.addEventListener('click', () => setMode('origin'));
  btnDest.addEventListener('click', () => setMode('dest'));

  // ----- Coord text input validate + apply -----
  function validateInput(el, min, max) {
    const v = parseFloat(el.value);
    if (isNaN(v) || v < min || v > max) {
      el.classList.add('invalid');
      return null;
    }
    el.classList.remove('invalid');
    return v;
  }
  btnApply.addEventListener('click', () => {
    const oLon = validateInput(document.getElementById('in-origin-lon'), -180, 180);
    const oLat = validateInput(document.getElementById('in-origin-lat'), -90, 90);
    const dLon = validateInput(document.getElementById('in-dest-lon'), -180, 180);
    const dLat = validateInput(document.getElementById('in-dest-lat'), -90, 90);
    if (oLon !== null && oLat !== null) setOrigin(L.latLng(oLat, oLon));
    if (dLon !== null && dLat !== null) setDest(L.latLng(dLat, dLon));
    if (originMarker && destMarker) {
      const grp = L.featureGroup([originMarker, destMarker]);
      map.fitBounds(grp.getBounds().pad(0.3));
    }
  });

  // ----- Add NGZ -----
  btnNgz.addEventListener('click', () => {
    setMode(null);
    new L.Draw.Polygon(map, {
      shapeOptions: {
        color: COLOR_NGZ,
        fillColor: COLOR_NGZ,
        weight: 2,
        fillOpacity: 0.25
      }
    }).enable();
  });

  // ----- Clear All -----
  btnClear.addEventListener('click', () => {
    if (originMarker) { map.removeLayer(originMarker); originMarker = null; }
    if (destMarker) { map.removeLayer(destMarker); destMarker = null; }
    ngzLayers.clearLayers();
    renderNgzList();
    if (routeLayer) { map.removeLayer(routeLayer); routeLayer = null; }
    setMode(null);
    document.getElementById('stat-status').textContent = '已清空';
    document.getElementById('stat-status').className = 'v';
    document.getElementById('stat-npts').textContent = '—';
    document.getElementById('stat-dist').textContent = '—';
    document.getElementById('stat-ngz').textContent = '0';
    document.getElementById('stat-error').textContent = '—';
    document.getElementById('stat-error').className = 'v';
    ['in-origin-lon','in-origin-lat','in-dest-lon','in-dest-lat'].forEach(id => {
      const el = document.getElementById(id);
      el.value = '';
      el.classList.remove('invalid');
    });
    clearLog();
    logPlaceholder();
  });

  map.on('click', e => {
    if (mode === 'origin') {
      setOrigin(e.latlng);
      setMode(null);
    } else if (mode === 'dest') {
      setDest(e.latlng);
      setMode(null);
    }
  });

  map.on(L.Draw.Event.CREATED, e => {
    ngzLayers.addLayer(e.layer);
    document.getElementById('stat-ngz').textContent = ngzLayers.getLayers().length;
    renderNgzList();
  });

  // ============= NGZ list =============
  const NGZ_DEFAULT_STYLE = { weight: 2, fillOpacity: 0.25 };
  const NGZ_HOVER_STYLE = { weight: 4, fillOpacity: 0.5 };
  const ngzListEl = document.getElementById('ngz-list');
  const ngzListSection = document.getElementById('ngz-list-section');
  const ngzListCount = document.getElementById('ngz-list-count');

  function removeNgz(layer) {
    ngzLayers.removeLayer(layer);
    document.getElementById('stat-ngz').textContent = ngzLayers.getLayers().length;
    renderNgzList();
  }

  function renderNgzList() {
    const layers = ngzLayers.getLayers();
    ngzListCount.textContent = `(${layers.length})`;
    if (layers.length === 0) {
      ngzListSection.hidden = true;
      ngzListEl.innerHTML = '';
      return;
    }
    ngzListSection.hidden = false;
    ngzListEl.innerHTML = '';
    layers.forEach((layer, i) => {
      const ring = layer.getLatLngs()[0];
      const row = document.createElement('div');
      row.className = 'ngz-list-item';
      row.innerHTML = `
        <span class="meta"><span class="idx">#${i + 1}</span>${ring.length} pts</span>
        <button class="btn-x" title="刪除此 NGZ">×</button>
      `;
      row.addEventListener('mouseenter', () => layer.setStyle(NGZ_HOVER_STYLE));
      row.addEventListener('mouseleave', () => layer.setStyle(NGZ_DEFAULT_STYLE));
      row.querySelector('.btn-x').addEventListener('click', (ev) => {
        ev.stopPropagation();
        removeNgz(layer);
      });
      ngzListEl.appendChild(row);
    });
  }

  // ============= Log panel =============
  const logOut = document.getElementById('log-output');
  function clearLog() { logOut.innerHTML = ''; }
  function logPlaceholder() {
    logOut.innerHTML = '<span class="placeholder">尚未執行；按 ▶ Run Route 後顯示 run_p2p 的 debug output</span>';
  }
  function logLineHTML(html) {
    logOut.insertAdjacentHTML('beforeend', html + '\n');
    logOut.scrollTop = logOut.scrollHeight;
  }

  document.getElementById('btn-log-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(logOut.innerText).then(() => {
      const btn = document.getElementById('btn-log-copy');
      const oldText = btn.textContent;
      btn.textContent = '✓ Copied';
      setTimeout(() => { btn.textContent = oldText; }, 1200);
    });
  });
  document.getElementById('btn-log-clear').addEventListener('click', () => {
    clearLog();
    logPlaceholder();
  });

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ============= Antimeridian unwrap =============
  // 路徑跨 ±180° 經線時，Leaflet 預設會把它拉成水平橫切。
  // 對 lon 做累積 unwrap、讓相鄰點 lon 差永遠在 ±180 內，Leaflet 會畫出單條連續線。
  // 參考 routing_map/metrics.py:_unwrap_lon、routing_map/viz_layers.py:add_path_layer。
  function unwrapLon(lon, ref) {
    while (lon - ref > 180) lon -= 360;
    while (lon - ref < -180) lon += 360;
    return lon;
  }
  function unwrapPath(pathLL) {
    if (!pathLL || pathLL.length === 0) return [];
    const out = [];
    let ref = pathLL[0][0];
    for (const [lon, lat] of pathLL) {
      const u = unwrapLon(lon, ref);
      out.push([u, lat]);
      ref = u;
    }
    return out;
  }
  // 把任意 lng 折回 [-180, 180)；marker 可以在 unwrap 世界副本，但對 server / input 欄要正規化。
  function canonLon(lng) {
    return ((lng + 180) % 360 + 360) % 360 - 180;
  }
  // route 畫完後把 origin / dest marker 搬到 unwrapped 兩端，避免「marker 一邊、route 一邊」的視覺脫節。
  function syncMarkersToPath(unwrapped) {
    if (!unwrapped || unwrapped.length === 0) return;
    const first = unwrapped[0];
    const last = unwrapped[unwrapped.length - 1];
    if (originMarker) originMarker.setLatLng([first[1], first[0]]);
    if (destMarker)   destMarker.setLatLng([last[1], last[0]]);
  }

  // Sutherland-Hodgman 半平面 clip：留 lng 在 cutLng 同側的 ring。
  function clipPolygonToSide(poly, side, cutLng) {
    const out = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const cur = poly[i];
      const nxt = poly[(i + 1) % n];
      const curIn = side === 'lt' ? cur.lng <= cutLng : cur.lng >= cutLng;
      const nxtIn = side === 'lt' ? nxt.lng <= cutLng : nxt.lng >= cutLng;
      if (curIn) out.push(cur);
      if (curIn !== nxtIn) {
        const t = (cutLng - cur.lng) / (nxt.lng - cur.lng);
        out.push({ lat: cur.lat + t * (nxt.lat - cur.lat), lng: cutLng });
      }
    }
    return out;
  }

  // 處理跨 dateline 的 NGZ：偵測後切成兩塊 canonical polygons。
  // 不跨 dateline 的 polygon 直接回一個。回傳 [[lng, lat], ...] 陣列陣列。
  // 理由：Shapely 拿 vertices [-175, 175, ...] 會誤判成 lng -175→175 的大環帶（350° 寬），
  //       routing_map.ngz.split_polygon_at_antimeridian 雖然存在但實測對此 case 沒能阻擋路徑，
  //       前端先切好讓 server 拿到的永遠是乾淨小盒子最穩。
  function splitNgzAcrossDateline(ring) {
    if (!ring || ring.length < 3) return [];
    let ref = ring[0].lng;
    const unwrapped = ring.map(p => {
      let lng = p.lng;
      while (lng - ref > 180) lng -= 360;
      while (lng - ref < -180) lng += 360;
      ref = lng;
      return { lat: p.lat, lng };
    });
    let minLng = Infinity, maxLng = -Infinity;
    for (const p of unwrapped) {
      if (p.lng < minLng) minLng = p.lng;
      if (p.lng > maxLng) maxLng = p.lng;
    }
    if (minLng >= -180 && maxLng <= 180) {
      return [unwrapped.map(p => [p.lng, p.lat])];
    }
    const cutLng = maxLng > 180 ? 180 : -180;
    const left = clipPolygonToSide(unwrapped, 'lt', cutLng);
    const right = clipPolygonToSide(unwrapped, 'gt', cutLng);
    const shift = cutLng === 180 ? -360 : 360;
    const farSide = cutLng === 180 ? right : left;
    const nearSide = cutLng === 180 ? left : right;
    const shifted = farSide.map(p => ({ lat: p.lat, lng: p.lng + shift }));
    const pieces = [];
    if (nearSide.length >= 3) pieces.push(nearSide.map(p => [p.lng, p.lat]));
    if (shifted.length >= 3) pieces.push(shifted.map(p => [p.lng, p.lat]));
    return pieces;
  }

  function friendlyError(errStr) {
    if (!errStr) return '';
    if (errStr.startsWith('ngz_patch_unreachable')) return '路徑被 NGZ 阻擋，無法繞行';
    return errStr.split('\n')[0];
  }

  function renderLog(logStr) {
    clearLog();
    const lines = (logStr || '').split('\n');
    for (const line of lines) {
      let html = escapeHtml(line);
      html = html.replace(/(\[pipeline\]\[\w+\])/g, '<span class="tag">$1</span>');
      html = html.replace(/(\[server\])/g, '<span class="dim">$1</span>');
      html = html.replace(/(=)(-?\d+(?:\.\d+)?)/g, '$1<span class="num">$2</span>');
      if (line.includes('✓')) html = `<span class="ok">${html}</span>`;
      else if (line.includes('✗') || /Error|Exception|Traceback/i.test(line)) html = `<span class="err">${html}</span>`;
      else if (/warn|warning|fallback/i.test(line)) html = `<span class="warn">${html}</span>`;
      logLineHTML(html);
    }
  }

  // ============= Run Route (real fetch) =============
  btnRun.addEventListener('click', async () => {
    const statusEl = document.getElementById('stat-status');
    const errEl = document.getElementById('stat-error');
    if (!originMarker || !destMarker) {
      statusEl.textContent = '請先設 Origin 與 Dest';
      statusEl.className = 'v err';
      return;
    }
    statusEl.textContent = '執行中...';
    statusEl.className = 'v';
    errEl.textContent = '—';
    errEl.className = 'v';
    clearLog();
    logLineHTML('<span class="dim">--- POST /api/route ---</span>');

    const oLL = originMarker.getLatLng();
    const dLL = destMarker.getLatLng();
    const ngzs = [];
    ngzLayers.getLayers().forEach(layer => {
      const ring = layer.getLatLngs()[0];
      const pieces = splitNgzAcrossDateline(ring);
      for (const piece of pieces) {
        ngzs.push({ points: piece });
      }
    });

    const body = {
      origin: { lon: canonLon(oLL.lng), lat: oLL.lat },
      dest:   { lon: canonLon(dLL.lng), lat: dLL.lat },
      ngz_polygons: ngzs,
      ngz_mode: document.getElementById('ngz-route-mode').value
    };

    try {
      const r = await fetch('/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const j = await r.json();

      renderLog(j.log);

      if (routeLayer) { map.removeLayer(routeLayer); routeLayer = null; }

      if (j.error) {
        statusEl.textContent = '✗ Error';
        statusEl.className = 'v err';
        errEl.textContent = friendlyError(j.error);
        errEl.className = 'v err';
        document.getElementById('stat-npts').textContent = '—';
        document.getElementById('stat-dist').textContent = '—';
        // patching 失敗時把 baseline 用紅色虛線畫出來，讓使用者看到 NGZ 衝突在哪一段
        if (j.path_raw && j.path_raw.length > 0) {
          const unwrapped = unwrapPath(j.path_raw);
          routeLayer = L.polyline(
            unwrapped.map(([lon, lat]) => [lat, lon]),
            { color: '#d32f2f', weight: 3, opacity: 0.7, dashArray: '8,6' }
          ).addTo(map);
          syncMarkersToPath(unwrapped);
        }
      } else {
        const unwrapped = unwrapPath(j.path_final);
        routeLayer = L.polyline(
          unwrapped.map(([lon, lat]) => [lat, lon]),
          { color: COLOR_ROUTE, weight: 4, opacity: 0.9 }
        ).addTo(map);
        syncMarkersToPath(unwrapped);

        statusEl.textContent = '✓ 完成';
        statusEl.className = 'v ok';
        document.getElementById('stat-npts').textContent = j.n_points;
        document.getElementById('stat-dist').textContent = j.length_km != null ? j.length_km.toFixed(1) : '—';
        errEl.textContent = 'None';
        errEl.className = 'v ok';
      }
    } catch (e) {
      statusEl.textContent = '✗ Network error';
      statusEl.className = 'v err';
      errEl.textContent = String(e);
      errEl.className = 'v err';
    }
  });

  // ============= Init: 從 /api/init 拉 bbox center 對焦 =============
  async function initView() {
    try {
      const r = await fetch('/api/init');
      if (!r.ok) throw new Error(`init failed: ${r.status}`);
      const j = await r.json();
      map.setView([j.center[1], j.center[0]], j.zoom);
    } catch (e) {
      console.warn('init failed, fallback view', e);
    }
  }
  initView();
  logPlaceholder();
})();
