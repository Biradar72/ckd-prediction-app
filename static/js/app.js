document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('patientForm');
  const btnSubmit = document.getElementById('btnSubmit');
  const resultPlaceholder = document.getElementById('resultPlaceholder');
  const resultContent = document.getElementById('resultContent');
  const outcomeBox = document.getElementById('outcomeBox');
  const outcomeMain = document.getElementById('outcomeMain');
  const outcomeTitle = document.getElementById('outcomeTitle');
  const outcomeConfidence = document.getElementById('outcomeConfidence');
  const riskBadge = document.getElementById('riskBadge');
  const meterFill = document.getElementById('meterFill');
  const meterPercent = document.getElementById('meterPercent');
  const insightsList = document.getElementById('insightsList');
  const ruleBox = document.getElementById('ruleBox');
  const ruleAssessmentText = document.getElementById('ruleAssessmentText');
  const rawJsonResponse = document.getElementById('rawJsonResponse');
  const btnToggleJson = document.getElementById('btnToggleJson');

  const tabML = document.getElementById('tabML');
  const tabRule = document.getElementById('tabRule');
  const btnSampleHealthy = document.getElementById('btnSampleHealthy');
  const btnSampleAtRisk = document.getElementById('btnSampleAtRisk');
  const btnReset = document.getElementById('btnReset');
  const csvFileInput = document.getElementById('csvFileInput');
  const batchResults = document.getElementById('batchResults');

  let currentEngine = 'ml';

  // Toggle Mode
  tabML.addEventListener('click', () => {
    tabML.classList.add('active');
    tabRule.classList.remove('active');
    currentEngine = 'ml';
    btnSubmit.innerHTML = `<ion-icon name="pulse-outline"></ion-icon> Run Clinical Prediction (ML)`;
  });

  tabRule.addEventListener('click', () => {
    tabRule.classList.add('active');
    tabML.classList.remove('active');
    currentEngine = 'rule';
    btnSubmit.innerHTML = `<ion-icon name="shield-checkmark-outline"></ion-icon> Run WHO Rule Assessment`;
  });

  // Extract form data as numbers
  function getFormData() {
    const formData = new FormData(form);
    const data = {};
    for (const [key, value] of formData.entries()) {
      data[key] = parseFloat(value);
    }
    return data;
  }

  // Populate form with sample
  function populateForm(sample) {
    for (const [key, val] of Object.entries(sample)) {
      const el = document.getElementById(key);
      if (el) el.value = val;
    }
  }

  // Load sample records from API
  btnSampleHealthy.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sample/healthy');
      const data = await res.json();
      populateForm(data);
    } catch (e) {
      console.error('Failed to load sample', e);
    }
  });

  btnSampleAtRisk.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sample/at_risk');
      const data = await res.json();
      populateForm(data);
    } catch (e) {
      console.error('Failed to load sample', e);
    }
  });

  btnReset.addEventListener('click', () => {
    form.reset();
    resultPlaceholder.style.display = 'block';
    resultContent.style.display = 'none';
  });

  // Toggle JSON response
  btnToggleJson.addEventListener('click', () => {
    if (rawJsonResponse.style.display === 'none') {
      rawJsonResponse.style.display = 'block';
    } else {
      rawJsonResponse.style.display = 'none';
    }
  });

  // Form Submit Handler
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const data = getFormData();
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = `<ion-icon name="hourglass-outline"></ion-icon> Evaluating Biomarkers...`;

    try {
      if (currentEngine === 'ml') {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();

        displayMLResults(result);
      } else {
        const res = await fetch('/api/rule-based', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();

        displayRuleResults(result);
      }
    } catch (err) {
      alert(`Diagnostic API Error: ${err.message}`);
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.innerHTML = currentEngine === 'ml' 
        ? `<ion-icon name="pulse-outline"></ion-icon> Run Clinical Prediction`
        : `<ion-icon name="shield-checkmark-outline"></ion-icon> Run WHO Rule Assessment`;
    }
  });

  function displayMLResults(data) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';
    ruleBox.style.display = 'none';

    rawJsonResponse.textContent = JSON.stringify(data, null, 2);

    const isCKD = data.is_ckd;
    const probPercent = Math.round(data.ckd_probability * 1000) / 10;

    outcomeMain.textContent = data.prediction;
    outcomeTitle.textContent = "XGBoost ML Diagnostic Result";
    outcomeConfidence.textContent = `Confidence: ${data.confidence_percentage}% (${isCKD ? 'High Probability' : 'Negative for CKD'})`;

    if (isCKD) {
      outcomeBox.className = 'outcome-box positive';
      riskBadge.className = 'risk-badge high';
      riskBadge.textContent = `${data.risk_level} Risk`;
    } else {
      outcomeBox.className = 'outcome-box negative';
      riskBadge.className = 'risk-badge low';
      riskBadge.textContent = `${data.risk_level} Risk`;
    }

    meterPercent.textContent = `${probPercent}%`;
    meterFill.style.width = `${Math.min(Math.max(probPercent, 4), 100)}%`;

    // Update Insights List
    insightsList.innerHTML = '';
    if (data.biomarker_insights && data.biomarker_insights.length > 0) {
      data.biomarker_insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightsList.appendChild(li);
      });
    }
  }

  function displayRuleResults(data) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';
    ruleBox.style.display = 'block';

    rawJsonResponse.textContent = JSON.stringify(data, null, 2);

    outcomeTitle.textContent = "WHO Expert Logic Result";
    outcomeMain.textContent = data.assessment;
    outcomeConfidence.textContent = "Deterministic Clinical Decision Tree";

    const isHigh = data.assessment.toLowerCase().includes('high') || data.assessment.toLowerCase().includes('possible');
    outcomeBox.className = isHigh ? 'outcome-box positive' : 'outcome-box negative';
    riskBadge.className = isHigh ? 'risk-badge high' : 'risk-badge low';
    riskBadge.textContent = data.assessment;

    ruleAssessmentText.textContent = data.assessment;

    // Set estimated scale
    const fakePercent = isHigh ? 85 : 12;
    meterPercent.textContent = `${fakePercent}% (Rule Estimator)`;
    meterFill.style.width = `${fakePercent}%`;

    insightsList.innerHTML = '';
    data.biomarker_insights.forEach(insight => {
      const li = document.createElement('li');
      li.textContent = insight;
      insightsList.appendChild(li);
    });
  }

  // Batch CSV Upload Handler
  csvFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    batchResults.style.display = 'block';
    batchResults.innerHTML = `<p style="color: var(--primary);">Processing cohort CSV file via Batch API...</p>`;

    try {
      const res = await fetch('/api/predict/batch', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || err.error || 'Batch processing failed');
      }

      const resData = await res.json();
      renderBatchSummary(resData);
    } catch (err) {
      batchResults.innerHTML = `<p style="color: #f87171;">⚠️ ${err.message}</p>`;
    }
  });

  function renderBatchSummary(data) {
    let html = `
      <div style="margin-bottom: 0.75rem;">
        <strong>Batch Evaluation Complete:</strong> Analyzed ${data.total_records} records.<br>
        <span style="color: #f87171;">CKD Positive: ${data.ckd_positive_count}</span> | 
        <span style="color: #34d399;">CKD Negative: ${data.ckd_negative_count}</span>
      </div>
      <table style="width: 100%; text-align: left; font-size: 0.75rem; border-collapse: collapse;">
        <thead>
          <tr style="border-bottom: 1px solid var(--border);">
            <th style="padding: 4px;">#</th>
            <th style="padding: 4px;">Prediction</th>
            <th style="padding: 4px;">Probability</th>
            <th style="padding: 4px;">Risk Level</th>
          </tr>
        </thead>
        <tbody>
    `;

    data.results.slice(0, 8).forEach(r => {
      const color = r.is_ckd ? '#f87171' : '#34d399';
      html += `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
          <td style="padding: 4px;">${r.patient_index}</td>
          <td style="padding: 4px; color: ${color}; font-weight: 600;">${r.prediction}</td>
          <td style="padding: 4px;">${Math.round(r.ckd_probability * 100)}%</td>
          <td style="padding: 4px;">${r.risk_level}</td>
        </tr>
      `;
    });

    if (data.results.length > 8) {
      html += `<tr><td colspan="4" style="padding: 4px; color: var(--text-dim); text-align: center;">...and ${data.results.length - 8} more patients</td></tr>`;
    }

    html += `</tbody></table>`;
    batchResults.innerHTML = html;
  }
});

