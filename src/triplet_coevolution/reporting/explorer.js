(() => {
  'use strict';
  const get = id => document.getElementById(id);
  const data = JSON.parse(get('explore-data').textContent);
  document.documentElement.classList.add('js');
  const source = get('source');
  const search = get('search');
  const field = get('field');
  const minimum = get('minimum');
  const maximum = get('maximum');
  const label = get('label-filter');
  const status = get('table-status');
  let file = null;
  let scanNumber = 0;

  function fill(select, options) {
    select.replaceChildren(...options.map(([value, text]) => new Option(text, value)));
  }

  function current() {
    return data.tables[Number(source.value)] || {fields: [], rows: []};
  }

  function fieldMenu(fields) {
    fill(field, [['', 'Choose a field'], ...fields.map(name => [name, name])]);
    minimum.value = '';
    maximum.value = '';
  }

  function cells(tr, values, tag) {
    for (const value of values.slice(0, 100)) {
      const el = document.createElement(tag);
      const text = String(value);
      el.textContent = text === '' ? '—' : text.length > 400 ? text.slice(0, 399) + '…' : text;
      if (tag === 'th') el.scope = 'col';
      tr.append(el);
    }
  }

  function tableHeader(fields) {
    const head = get('head');
    head.replaceChildren();
    cells(head, fields, 'th');
    get('rows').replaceChildren();
  }

  function append(row) {
    const tr = document.createElement('tr');
    cells(tr, row, 'td');
    get('rows').append(tr);
  }

  function matches(fields, row) {
    const term = search.value.trim().toLowerCase();
    if (term && !row.some(value => String(value).toLowerCase().includes(term))) return false;
    const category = label.value.trim().toLowerCase();
    if (category && !fields.some((name, i) =>
      ['category', 'label_status', 'contact', 'status'].includes(name) &&
      String(row[i]).toLowerCase() === category)) return false;
    if (minimum.value === '' && maximum.value === '') return true;
    const value = row[fields.indexOf(field.value)];
    if (value == null || value === '' || !Number.isFinite(Number(value))) return false;
    return (minimum.value === '' || Number(value) >= Number(minimum.value)) &&
      (maximum.value === '' || Number(value) <= Number(maximum.value));
  }

  function preview() {
    scanNumber++;
    file = null;
    get('local-file').value = '';
    const table = current();
    tableHeader(table.fields);
    const filtered = table.rows.filter(row => matches(table.fields, row));
    filtered.slice(0, 100).forEach(append);
    status.textContent = 'Showing ' + Math.min(100, filtered.length) + ' of ' + filtered.length +
      ' matching preview rows. Preview contains ' + table.rows.length + ' rows from ' +
      (table.title || 'this table') + '.';
  }

  // Parse records incrementally. The quote state survives a chunk boundary in
  // the middle of a quoted newline or escaped double quote.
  async function scan(chosen, reset) {
    const id = ++scanNumber;
    file = chosen;
    const separator = chosen.name.toLowerCase().endsWith('.csv') ? ',' : '\t';
    let fields = null, record = [], cell = '';
    let quoted = false, afterQuote = false, carriageReturn = false, quotedField = false;
    let total = 0, hits = 0, drawn = 0, size = 0;
    function finish() {
      record.push(cell);
      cell = '';
      size = 0;
      const hasRecord = quotedField || record.length > 1 || record[0] !== '';
      quotedField = false;
      if (!hasRecord) {
        record = [];
        return;
      }
      if (!fields) {
        fields = record;
        if (fields.length > 512) throw Error('The local reader supports up to 512 fields.');
        fields[0] = fields[0].replace(/^\uFEFF/, '');
        if (reset) fieldMenu(fields);
        tableHeader(fields);
      } else {
        if (record.length !== fields.length) throw Error('A row has a different field count from its header.');
        total++;
        if (matches(fields, record)) {
          hits++;
          if (drawn < 100) {
            append(record);
            drawn++;
          }
        }
      }
      record = [];
    }
    let reader;
    status.textContent = 'Reading ' + chosen.name + ' locally…';
    try {
      reader = chosen.stream().pipeThrough(new TextDecoderStream()).getReader();
      while (true) {
        const part = await reader.read();
        if (id !== scanNumber) {
          await reader.cancel();
          return;
        }
        if (part.done) break;
        for (const char of part.value) {
          if (++size > 4194304) throw Error('A local table record exceeds 4,194,304 decoded characters.');
          if (carriageReturn) {
            carriageReturn = false;
            if (char === '\n') continue;
          }
          if (afterQuote) {
            afterQuote = false;
            if (char === '"') {
              cell += '"';
              quoted = true;
              continue;
            }
            if (![separator, '\n', '\r'].includes(char)) throw Error('Invalid text after a quoted field.');
          }
          if (quoted) {
            if (char === '"') {
              quoted = false;
              afterQuote = true;
            } else {
              cell += char;
            }
          } else if (char === '"') {
            if (cell !== '') throw Error('Unexpected quote in an unquoted field.');
            quoted = true;
            quotedField = true;
          } else if (char === separator) {
            record.push(cell);
            if (record.length >= 512) throw Error('The local reader supports up to 512 fields.');
            cell = '';
          } else if (char === '\r' || char === '\n') {
            finish();
            carriageReturn = char === '\r';
          } else {
            cell += char;
          }
        }
      }
      if (quoted) throw Error('Unclosed quoted field.');
      if (cell || record.length || afterQuote) finish();
      if (!fields) tableHeader([]);
      status.textContent = 'Showing ' + drawn + ' of ' + hits + ' matching rows across ' +
        total + ' rows in ' + chosen.name + '.';
    } catch (error) {
      if (id === scanNumber) {
        tableHeader([]);
        status.textContent = 'Could not read this table: ' + error.message;
      }
    } finally {
      if (reader) reader.releaseLock();
    }
  }

  fill(source, data.tables.map((table, i) => [i, table.title]));
  source.onchange = () => {
    fieldMenu(current().fields);
    preview();
  };
  get('apply').onclick = () => file ? scan(file, false) : preview();
  get('local-file').onchange = event => {
    if (event.target.files[0]) scan(event.target.files[0], true);
  };
  for (const el of [search, label, minimum, maximum]) {
    el.onkeydown = event => { if (event.key === 'Enter') get('apply').click(); };
  }
  get('palette').onchange = event => {
    for (const [type, ramp] of Object.entries(data.colors[event.target.value])) {
      ramp.forEach((color, i) => document.body.style.setProperty('--' + type + i, color));
    }
  };
  get('map-selection').onchange = event => {
    const choice = event.target.value;
    document.querySelectorAll('[data-score-map]').forEach(el => {
      el.hidden = Boolean(choice && el.dataset.scoreMap !== choice);
    });
  };
  fieldMenu(current().fields);
  preview();
})();
