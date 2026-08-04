// Interactive debt-payoff simulator. Reads the loan parameters from the data-*
// attributes on #payoff and lets the user drag the yearly payment to see how many
// years the debt takes to clear and how much is left each year.
// Used on both the Live Prediction page and the Customer detail page.
// Wrapped in an IIFE (a function that runs itself) so my variables don't leak onto
// the rest of the page.
(function () {
  var el = document.getElementById('payoff');
  if (!el) return;   // this page has no simulator (e.g. a rejected applicant) -> stop
  // read the loan details Flask put on the HTML element as data-* attributes.
  // the leading "+" converts the text values into numbers.
  var P = +el.dataset.principal, rate = +el.dataset.rate, income = +el.dataset.income,
      foir = +el.dataset.foir, emi = +el.dataset.emi, tenure = +el.dataset.tenure;
  var scheduledAnnual = emi * 12;             // what they'd pay in a year on the normal EMI
  var annualInterest0 = P * (rate / 100);     // first-year interest if nothing is repaid

  // work out sensible slider limits: never let the payment be so low it can't beat
  // the interest, and cap the top at 1.6x the normal payment. Rounded to nice 1000s.
  var minPay = Math.max(Math.round(annualInterest0 * 1.12 / 1000) * 1000,
                        Math.round(scheduledAnnual * 0.4 / 1000) * 1000);
  if (minPay >= scheduledAnnual) minPay = Math.round(scheduledAnnual * 0.7 / 1000) * 1000;
  var maxPay = Math.round(scheduledAnnual * 1.6 / 1000) * 1000;

  var slider = document.getElementById('pay-slider');
  // start the slider on the normal EMI amount, between the min and max we just found
  slider.min = minPay; slider.max = maxPay; slider.step = 1000; slider.value = scheduledAnnual;

  // format a number as Indian rupees, e.g. 219100 -> "Rs.2,19,100"
  function inr(x) {
    x = Math.round(x); var neg = x < 0; x = Math.abs(x); var s = '' + x;
    if (s.length > 3) { var last3 = s.slice(-3), rest = s.slice(0, -3), parts = [];
      while (rest.length > 2) { parts.unshift(rest.slice(-2)); rest = rest.slice(0, -2); }
      if (rest) parts.unshift(rest); s = parts.join(',') + ',' + last3; }
    return (neg ? '-Rs.' : 'Rs.') + s;
  }

  // The core maths: given how much they pay per year, step through month by month and
  // see how long it takes to clear the debt. Same amortization logic as the Python side.
  function simulate(annualPayment) {
    var rM = (rate / 100) / 12, pay = annualPayment / 12, bal = P, month = 0;
    var years = [], yP = 0, yI = 0, MAX = 40 * 12, totalInterest = 0;
    // if the monthly payment can't even cover one month's interest, it never clears
    if (pay <= bal * rM) return { clears: false };
    while (bal > 0 && month < MAX) {
      month++;
      var interest = bal * rM, principal = pay - interest;  // split payment into two parts
      if (principal > bal) principal = bal;                 // don't overpay the last month
      bal -= principal; yP += principal; yI += interest; totalInterest += interest;
      if (month % 12 === 0 || bal <= 0) {   // end of a year (or fully paid) -> save a row
        years.push({ year: Math.ceil(month / 12), principal: yP, interest: yI,
                     paid: yP + yI, remaining: Math.max(bal, 0) });
        yP = 0; yI = 0;                     // reset the yearly counters
      }
    }
    return { clears: bal <= 0, years: years, months: month, totalInterest: totalInterest };
  }

  // give the EMI% a colour zone -- same thresholds as the Python foir_zone().
  function zoneOf(pct) {
    if (pct <= 30) return { label: 'Ideal', cls: 'ideal' };
    if (pct <= 40) return { label: 'Moderate', cls: 'moderate' };
    if (pct <= 50) return { label: 'Caution', cls: 'caution' };
    return { label: 'High risk', cls: 'high' };
  }

  // redraw everything whenever the slider moves: the summary line + the year-by-year bars
  function render() {
    var annual = +slider.value;
    document.getElementById('pay-label').textContent = inr(annual) + ' / year';
    var emiPct = (annual / 12) / income * 100;
    var z = zoneOf(emiPct);
    var pe = document.getElementById('pay-emi');
    pe.innerHTML = '(' + (annual / 12 >= 1 ? inr(annual / 12) : 0) + '/mo · ' +
                   emiPct.toFixed(0) + '% of income · <span class="zone-tag ' + z.cls + '">' +
                   z.label + '</span>)';
    pe.className = 'muted';

    var sim = simulate(annual);
    var summary = document.getElementById('payoff-summary');
    var box = document.getElementById('payoff-years');
    if (!sim.clears) {
      summary.className = 'payoff-summary warn';
      summary.innerHTML = '&#9888; This payment is too low to ever clear the debt — it barely covers the interest.';
      box.innerHTML = ''; return;
    }
    var yrs = sim.years.length;
    var extra = yrs > tenure ? ' — ' + (yrs - tenure) + ' year(s) longer than the scheduled ' + tenure
               : (yrs < tenure ? ' — ' + (tenure - yrs) + ' year(s) faster than the scheduled ' + tenure : ' — same as the scheduled tenure');
    summary.className = 'payoff-summary ok';
    summary.innerHTML = '<b>Debt cleared in ' + yrs + ' year' + (yrs > 1 ? 's' : '') + '</b>' + extra +
      '. Total interest paid: <b>' + inr(sim.totalInterest) + '</b>. This payment sits in the ' +
      '<span class="zone-tag ' + z.cls + '">' + z.label + '</span> zone' +
      (emiPct > 30 ? ' — above the 30% ideal zone, so it starts eating into savings/needs.' : '.');

    var maxPaid = Math.max.apply(null, sim.years.map(function (y) { return y.paid; }));
    box.innerHTML = sim.years.map(function (y) {
      return '<div class="sched-year"><div class="sched-head"><span>Year ' + y.year +
        '</span><span class="muted">Debt left: ' + inr(y.remaining) + '</span></div>' +
        '<div class="stack-bar"><div class="seg principal" style="width:' + (y.principal / maxPaid * 100) +
        '%"></div><div class="seg interest" style="width:' + (y.interest / maxPaid * 100) + '%"></div></div>' +
        '<div class="sched-legend muted">Paid ' + inr(y.paid) + ' (principal ' + inr(y.principal) +
        ' + interest ' + inr(y.interest) + ')</div></div>';
    }).join('');
  }

  slider.addEventListener('input', render);   // re-run render() every time the slider moves
  render();                                    // draw once on page load
})();
