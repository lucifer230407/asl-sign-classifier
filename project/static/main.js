const predLetter = document.getElementById('predLetter');
const predConf   = document.getElementById('predConf');
const confBar    = document.getElementById('confBar');
const top3List   = document.getElementById('top3List');

function updateAlphabet(letter) {
  document.querySelectorAll('.alpha-cell').forEach(el => el.classList.remove('active'));
  if (letter) {
    const el = document.getElementById(`alpha-${letter}`);
    if (el) el.classList.add('active');
  }
}

function updateTop3(top3) {
  top3List.innerHTML = '';
  top3.forEach(([lbl, conf], i) => {
    const item = document.createElement('div');
    item.className = 'top3-item' + (i === 0 ? ' highlight' : '');
    item.innerHTML = `
      <span class="top3-label">${lbl}</span>
      <div class="top3-bar-wrap">
        <div class="top3-bar" style="width:${(conf * 100).toFixed(1)}%"></div>
      </div>
      <span class="top3-conf">${(conf * 100).toFixed(1)}%</span>
    `;
    top3List.appendChild(item);
  });
}

async function fetchPrediction() {
  try {
    const res  = await fetch('/prediction');
    const data = await res.json();
    const { letter, confidence, top3 } = data;
    const pct = (confidence * 100).toFixed(1);

    if (letter) {
      predLetter.textContent = letter;
      predLetter.className   = 'pred-letter';
    } else {
      predLetter.textContent = '?';
      predLetter.className   = 'pred-letter low';
    }

    predConf.textContent     = `Confidence: ${pct}%`;
    confBar.style.width      = `${pct}%`;
    confBar.style.background = confidence >= 0.9
      ? 'linear-gradient(90deg, #00c8ff, #00ffb2)'
      : 'linear-gradient(90deg, #ff4060, #ff8040)';

    updateAlphabet(letter);
    if (top3 && top3.length) updateTop3(top3);

  } catch (e) {
    // server not ready yet
  }
}

setInterval(fetchPrediction, 300);
fetchPrediction();
