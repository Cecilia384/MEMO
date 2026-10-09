'use strict';
const steps = [
  ['READ THE STREAM AT MULTIPLE LEVELS', 'Multi-Level Entity-Aware Perception', 'Measure semantic continuity through global semantics, local entity features, and spatial structure. Together, these signals capture both the scene and the details within it.'],
  ['GROUP MOMENTS THAT BELONG TOGETHER', 'Online Temporal Chunking', 'An adaptive similarity threshold identifies changes in the stream and forms semantically coherent chunks. Memory is organized around meaningful visual continuity.'],
  ['INDEX LIGHTLY. PRESERVE THE EVIDENCE.', 'Structured Memory Construction', 'Store lightweight global and entity-level retrieval indices on GPU, with high-resolution visual evidence on CPU. Each chunk connects compact search representations with its visual evidence.'],
  ['BRING THE RIGHT MOMENTS BACK', 'Query-Specific Evidence Retrieval', 'Use the query to recall evidence from relevant past chunks. Combine that evidence with frames from the current active segment so the MLLM can reason over both past and present.']
];
const stepButtons = [...document.querySelectorAll('[data-step]')];
function selectStep(index, focus = false) {
  stepButtons.forEach((button, i) => { button.setAttribute('aria-selected', String(i === index)); button.tabIndex = i === index ? 0 : -1; });
  const [kicker, title, description] = steps[index];
  document.querySelector('#step-kicker').textContent = kicker;
  document.querySelector('#step-title').textContent = title;
  document.querySelector('#step-description').textContent = description;
  document.querySelector('#method-panel').setAttribute('aria-labelledby', `step-${index}`);
  if (focus) stepButtons[index].focus();
}
stepButtons.forEach((button, index) => {
  button.addEventListener('click', () => selectStep(index));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % steps.length;
    if (event.key === 'ArrowLeft') next = (index + steps.length - 1) % steps.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = steps.length - 1;
    if (next !== undefined) { event.preventDefault(); selectStep(next, true); }
  });
});
const backbones = [
  {name:'LLaVA-OneVision-0.5B', ovo:[49.7,50.4], stream:[59.6,59.5]},
  {name:'LLaVA-OneVision-7B', ovo:[63.1,66.6], stream:[71.1,73.2]},
  {name:'Qwen2.5-VL-7B', ovo:[59.9,66.9], stream:[73.3,78.5]},
  {name:'Qwen3-VL-8B', ovo:[70.1,76.0], stream:[73.2,83.7]}
];
function renderChart(benchmark) {
  document.querySelectorAll('[data-benchmark]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.benchmark === benchmark)));
  document.querySelector('#chart-title').textContent = benchmark === 'stream' ? 'StreamingBench · real-time subset' : 'OVO-Bench · six real-time tasks';
  const fragment = document.createDocumentFragment();
  for (const model of backbones) {
    const [base,memo] = model[benchmark];
    const gain = (memo-base).toFixed(1);
    const row = document.createElement('div'); row.className = 'chart-row';
    const name = document.createElement('div'); name.className = 'model-name'; name.textContent = model.name;
    const delta = document.createElement('small'); delta.textContent = `${Number(gain) >= 0 ? '+' : '−'}${Math.abs(Number(gain)).toFixed(1)} pp with MEMO`; name.append(delta);
    const bars = document.createElement('div'); bars.className = 'bars';
    [base,memo].forEach((score,index) => {
      const line = document.createElement('div'); line.className = 'bar-line';
      line.setAttribute('aria-label', `${index ? 'With MEMO' : 'Backbone'}: ${score.toFixed(1)} percent`);
      const bar = document.createElement('div'); bar.className = `bar${index ? ' memo' : ''}`; bar.style.setProperty('--score', `${score}%`);
      const value = document.createElement('span'); value.className = 'bar-value'; value.textContent = score.toFixed(1); value.setAttribute('aria-hidden','true');
      bar.append(value); line.append(bar); bars.append(line);
    });
    row.append(name,bars); fragment.append(row);
  }
  document.querySelector('#results-chart').replaceChildren(fragment);
}
document.querySelectorAll('[data-benchmark]').forEach(button => button.addEventListener('click', () => renderChart(button.dataset.benchmark)));
renderChart('stream');
const rows = [
  ['group','Online methods with training'],
  ['VideoLLM-online-8B','2 fps',20.8,36.0],['Dispider-7B','1 fps',54.6,67.6],['Flash-VStream-7B','1 fps',29.9,23.2],['ViSpeak','1 fps',66.3,74.4],['TimeChat-Online-7B','1 fps',61.9,75.3],['StreamForest-7B','1 fps',61.2,77.3],
  ['group','Training-free online methods'],
  ['LLaVA-OneVision-0.5B','32',49.7,59.6],['↳ + ReKV','0.5 fps',43.8,57.4],['↳ + MEMO','1 fps',50.4,59.5],
  ['LLaVA-OneVision-7B','32',63.1,71.1],['↳ + ReKV','0.5 fps',57.3,69.1],['↳ + LiveVLM','0.5 fps',null,72.9],['↳ + StreamKV','0.5 fps',null,68.8],['↳ + MEMO','1 fps',66.6,73.2],
  ['Qwen2.5-VL-7B','1 fps',59.9,73.3],['↳ + FluxMem','1 fps',67.2,76.4],['↳ + MEMO','1 fps',66.9,78.5],
  ['Qwen3-VL-8B','1 fps',70.1,73.2],['↳ + MEMO','1 fps',76.0,83.7]
];
for (const data of rows) {
  const row = document.createElement('tr');
  if (data[0] === 'group') {row.className='group-row'; const heading=document.createElement('th'); heading.colSpan=4; heading.textContent=data[1]; row.append(heading);}
  else {if(data[0].includes('MEMO')) row.className='memo-row'; data.forEach((value,index)=>{const cell=document.createElement(index===0?'th':'td'); if(index===0)cell.scope='row'; cell.textContent=typeof value==='number'?value.toFixed(1):value??'—'; row.append(cell);});}
  document.querySelector('#full-results').append(row);
}
const menu = document.querySelector('.menu-toggle');
const links = document.querySelector('#nav-links');
function closeMenu(){menu.setAttribute('aria-expanded','false');links.classList.remove('is-open');}
menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));links.classList.toggle('is-open',open);});
links.querySelectorAll('a').forEach(link=>link.addEventListener('click',closeMenu));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&menu.getAttribute('aria-expanded')==='true'){closeMenu();menu.focus();}});
const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{const img=document.querySelector('#dialog-image');img.src=button.dataset.zoom;img.alt=button.querySelector('img').alt;document.querySelector('#dialog-caption').textContent=button.dataset.caption;dialog.showModal();}));
document.querySelector('#close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)dialog.close();}});
document.querySelector('#copy-citation').addEventListener('click',async()=>{
  const code=document.querySelector('#bibtex');
  try {await navigator.clipboard.writeText(code.textContent);document.querySelector('#copy-status').textContent='Citation copied to clipboard.';}
  catch {const range=document.createRange();range.selectNodeContents(code);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);document.querySelector('#copy-status').textContent='Citation selected. Press Ctrl+C (or ⌘C on Mac) to copy.';}
});
