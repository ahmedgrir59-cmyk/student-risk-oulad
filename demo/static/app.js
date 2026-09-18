const $ = (id)=>document.getElementById(id);
let selectedRow = null;
const pct = (v,d=1)=> v==null?'—':`${(Number(v)*100).toFixed(d)}%`;
const num = (v,d=0)=> v==null?'—':Number(v).toFixed(d);
const esc = (s)=>String(s??'').replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]));
const featureLabels={
  assessment_weight_completed_28d:'Assessment weight completed',
  weighted_assessment_score_28d:'Weighted assessment score',
  active_days_28d:'Active days',
  assessment_weight_due_28d:'Assessment weight due',
  unique_sites_28d:'Unique VLE sites',
  assessment_completion_rate_28d:'Assessment completion rate',
  code_module:'Module',
  late_submissions_28d:'Late submissions',
  highest_education:'Highest education',
  assessments_submitted_28d:'Assessments submitted',
  assessments_missed_28d:'Missed assessments',
  clicks_last7d:'Clicks in last 7 days',
  active_days_last7d:'Active days in last 7 days',
  days_since_last_activity_28d:'Days since last activity',
  weekly_click_slope_28d:'Weekly engagement trend',
  total_clicks_28d:'Total VLE clicks'
};
const prettyFeature=(name)=>featureLabels[name]||String(name||'').replace(/_28d$/,'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
const originalValue=(v)=>v==null?'missing':(typeof v==='number'?(Number.isInteger(v)?String(v):Number(v).toFixed(2)):String(v));
function showError(message){ const b=$('errorBanner'); b.textContent=message; b.hidden=!message; }
async function api(path){
  let r;
  try { r=await fetch(path,{headers:{'Accept':'application/json'}}); }
  catch(e){ throw new Error('Dashboard server is not reachable.'); }
  let j={};
  try { j=await r.json(); } catch(e){ throw new Error(`Invalid server response (${r.status}).`); }
  if(!r.ok) throw new Error(j.message||`Request failed (${r.status})`);
  return j;
}

async function loadSummary(){
  const s=await api('/api/summary');
  $('totalRows').textContent=s.totalRows.toLocaleString();
  $('uniqueStudents').textContent=s.totalStudents.toLocaleString();
  $('highCount').textContent=(s.riskCounts.High||0).toLocaleString();
  $('mediumCount').textContent=(s.riskCounts.Medium||0).toLocaleString();
  $('lowCount').textContent=(s.riskCounts.Low||0).toLocaleString();
  $('highRate').textContent=`Observed risk: ${pct(s.riskActualRates.High)}`;
  $('mediumRate').textContent=`Observed risk: ${pct(s.riskActualRates.Medium)}`;
  $('lowRate').textContent=`Observed risk: ${pct(s.riskActualRates.Low)}`;
  $('recall').textContent=pct(s.model.recall);
  $('rocAuc').textContent=s.model.rocAuc.toFixed(3);
  $('modelName').textContent=`${s.model.name} · Day 28`;
}
async function loadStudents(){
  const risk=$('riskFilter').value, search=$('searchInput').value.trim(), sort=$('sortFilter').value;
  const rows=await api(`/api/students?risk=${encodeURIComponent(risk)}&search=${encodeURIComponent(search)}&sort=${encodeURIComponent(sort)}&limit=120`);
  $('studentCountBadge').textContent=`${rows.length} shown`;
  $('studentList').innerHTML=rows.map(r=>`<button type="button" class="student-item ${selectedRow===r.row_index?'active':''}" data-row="${r.row_index}" aria-pressed="${selectedRow===r.row_index?'true':'false'}">
    <span><span class="student-id">Student ${esc(r.id_student)}</span><span class="student-sub">${esc(r.code_module)} · ${esc(r.code_presentation)} · ${esc(r.engagement_direction||'')}</span></span>
    <span><span class="mini-risk ${esc(r.risk_level)}">${esc(r.risk_level)} ${pct(r.risk_probability,0)}</span></span>
  </button>`).join('') || '<p class="empty">No students match the filters.</p>';
  document.querySelectorAll('.student-item').forEach(el=>el.addEventListener('click',()=>selectStudent(Number(el.dataset.row))));
  if(selectedRow==null && rows.length) await selectStudent(rows[0].row_index,{refreshList:false});
}
function drawTrend(values){
  const svg=$('trendChart'); const w=640,h=250,pad=45; const max=Math.max(...values,1)*1.12;
  const xs=values.map((_,i)=>pad+i*(w-2*pad)/3); const ys=values.map(v=>h-pad-(v/max)*(h-2*pad));
  let grid=''; for(let i=0;i<4;i++){const y=pad+i*(h-2*pad)/3; grid+=`<line x1="${pad}" y1="${y}" x2="${w-pad}" y2="${y}" stroke="#e9edf4"/>`;}
  const pts=xs.map((x,i)=>`${x},${ys[i]}`).join(' '); const area=`${pad},${h-pad} ${pts} ${w-pad},${h-pad}`;
  svg.setAttribute('aria-label',`Weekly VLE clicks: Week 1 ${Math.round(values[0])}, Week 2 ${Math.round(values[1])}, Week 3 ${Math.round(values[2])}, Week 4 ${Math.round(values[3])}`);
  svg.innerHTML=`${grid}<polygon points="${area}" fill="rgba(55,102,232,.08)"/><polyline points="${pts}" fill="none" stroke="#3766e8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  ${values.map((v,i)=>`<circle cx="${xs[i]}" cy="${ys[i]}" r="6" fill="#fff" stroke="#3766e8" stroke-width="3"/><text x="${xs[i]}" y="${ys[i]-13}" text-anchor="middle" font-size="13" font-weight="800" fill="#24314b">${Math.round(v)}</text><text x="${xs[i]}" y="${h-17}" text-anchor="middle" font-size="11" fill="#7a8498">Week ${i+1}</text>`).join('')}`;
}
function renderRecommendations(recs){
  $('recommendations').innerHTML=(recs||[]).map(r=>`<div class="rec"><strong>${esc(r.recommendation)}</strong><p>${esc(r.reason)}</p></div>`).join('') || '<p class="empty">No recommendation rule was triggered.</p>';
}
function renderLocal(ex){
  if(!ex){$('localFactors').innerHTML='<p class="empty">Local explanation is unavailable for this enrollment.</p>';return;}
  const group=(title,items,cls)=>`<div class="factor-group"><div class="factor-title">${title}</div>${items.map(x=>`<div class="factor ${cls}" title="${esc(x.transformed_feature||x.source_feature)}"><div class="factor-name">${esc(prettyFeature(x.source_feature))} <span class="factor-original">value: ${esc(originalValue(x.original_value))}</span></div><div class="factor-val">${x.shap_value>0?'+':''}${Number(x.shap_value).toFixed(3)}</div></div>`).join('')}</div>`;
  $('localFactors').innerHTML='<div class="explain-chip">Computed for this enrollment</div>'+group('Raises the model risk score',ex.risk||[],'risk-factor')+group('Lowers the model risk score',ex.protective||[],'protect-factor')+'<p class="note">Signed SHAP values explain the base Logistic Regression score for this selected enrollment before probability calibration. Positive values raise the modeled risk score; negative values lower it. They do not prove causality.</p>';
}
async function selectStudent(row,{refreshList=true}={}){
  selectedRow=Number(row); if(refreshList) await loadStudents();
  const s=await api(`/api/student?row_index=${selectedRow}`);
  $('studentTitle').textContent=`Student ${s.id_student}`; $('studentMeta').textContent=`${s.code_module} · ${s.code_presentation} · row ${s.row_index}`;
  $('riskPill').className=`risk-pill ${s.risk_level}`; $('riskPill').textContent=`${s.risk_level} risk`; $('riskProbability').textContent=pct(s.risk_probability);
  const riskPct=Math.min(100,Math.max(0,s.risk_probability*100)); $('gaugeFill').style.width=`${riskPct}%`; document.querySelector('.gauge').setAttribute('aria-valuenow',riskPct.toFixed(1));
  $('engagementBadge').textContent=s.engagement_direction||'—'; drawTrend([s.clicks_week1_28d||0,s.clicks_week2_28d||0,s.clicks_week3_28d||0,s.clicks_week4_28d||0]);
  $('activeDays').textContent=num(s.active_days_28d); $('active7').textContent=num(s.active_days_last7d); $('daysSince').textContent=num(s.days_since_last_activity_28d);
  $('assessmentScore').textContent=s.assessment_mean_score_28d==null?'No early score':num(s.assessment_mean_score_28d,1); $('completionRate').textContent=s.assessment_completion_rate_28d==null?'—':pct(s.assessment_completion_rate_28d,0); $('missedAssessments').textContent=num(s.assessments_missed_28d);
  renderRecommendations(s.recommendations); renderLocal(s.localExplanation); $('livePredictBtn').disabled=false; $('liveStatus').textContent='';
}
async function livePredict(){
  if(selectedRow==null)return; $('livePredictBtn').disabled=true; $('liveStatus').textContent='Running saved preprocessing + calibrated model…';
  try{const p=await api(`/api/predict?row_index=${selectedRow}`); $('liveStatus').textContent=`Live inference confirmed: ${p.risk_level} · ${pct(p.risk_probability)} at Day ${p.prediction_point_day}`; renderRecommendations(p.recommendations);}
  catch(e){$('liveStatus').textContent=`Inference error: ${e.message}`;} finally{$('livePredictBtn').disabled=false;}
}
async function loadImportance(){
  const rows=await api('/api/global-importance'); const max=Math.max(...rows.map(r=>r.importance),.001);
  $('importanceBars').innerHTML=rows.map(r=>`<div class="importance-row" title="${esc(r.feature)}"><div class="importance-name">${esc(prettyFeature(r.feature))}</div><div class="bar-track"><div class="bar-fill" style="width:${r.importance/max*100}%"></div></div><div class="importance-value">${r.importance.toFixed(3)}</div></div>`).join('');
}
async function loadModels(){
  const rows=await api('/api/models'); const max=Math.max(...rows.map(r=>r.f1_mean),.001);
  $('modelComparison').innerHTML=rows.map(r=>`<div class="model-row ${Number(r.selection_rank)===1?'selected-model':''}"><div class="model-row-head"><span>${esc(r.model.replaceAll('_',' ').replace('linear svm','Linear SVM'))}${Number(r.selection_rank)===1?' <em class="selected-tag">Selected</em>':''}</span><span>F1 ${r.f1_mean.toFixed(3)}</span></div><div class="bar-track"><div class="bar-fill" style="width:${r.f1_mean/max*100}%"></div></div><div class="model-sub">Recall ${r.recall_mean.toFixed(3)} · ROC-AUC ${r.roc_auc_mean.toFixed(3)}</div></div>`).join('');
}
async function loadFairness(){
  const rows=await api('/api/fairness');
  const preferred=[['disability','Y'],['disability','N'],['gender','F'],['gender','M']];
  const selected=preferred.map(([a,g])=>rows.find(r=>String(r.attribute)===a&&String(r.group)===g)).filter(Boolean);
  $('fairnessSnapshot').innerHTML=selected.map(r=>`<div class="fairness-row"><div><strong>${esc(r.attribute)}: ${esc(r.group)}</strong><span>n=${Number(r.n).toLocaleString()}</span></div><div class="fairness-value">FPR ${pct(r.false_positive_rate)}</div></div>`).join('') || '<p class="empty">Fairness audit rows unavailable.</p>';
}
let timer;
$('searchInput').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>loadStudents().catch(e=>showError(e.message)),220)});
$('riskFilter').addEventListener('change',()=>loadStudents().catch(e=>showError(e.message)));
$('sortFilter').addEventListener('change',()=>loadStudents().catch(e=>showError(e.message)));
$('livePredictBtn').addEventListener('click',()=>livePredict().catch(e=>showError(e.message)));
Promise.all([loadSummary(),loadImportance(),loadModels(),loadFairness(),api('/api/health')])
  .then(()=>loadStudents())
  .then(()=>showError(''))
  .catch(e=>{console.error(e);showError(e.message);});
