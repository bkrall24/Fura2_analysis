const css = `
.bk-root .bk-input {
  display: inline-block;
  width: 100%;
  flex-grow: 1;
  -webkit-flex-grow: 1;
  min-height: 31px;
  padding: 0 12px;
  background-color: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.bk-root .bk-input:focus {
  border-color: #66afe9;
  outline: 0;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.bk-root .bk-input::placeholder,
.bk-root .bk-input:-ms-input-placeholder,
.bk-root .bk-input::-moz-placeholder,
.bk-root .bk-input::-webkit-input-placeholder {
  color: #999;
  opacity: 1;
}
.bk-root .bk-input[disabled] {
  cursor: not-allowed;
  background-color: #eee;
  opacity: 1;
}
.bk-root select:not([multiple]).bk-input,
.bk-root select:not([size]).bk-input {
  height: auto;
  appearance: none;
  -webkit-appearance: none;
  background-image: url('data:image/svg+xml;utf8,<svg version="1.1" viewBox="0 0 25 20" xmlns="http://www.w3.org/2000/svg"><path d="M 0,0 25,0 12.5,20 Z" fill="black" /></svg>');
  background-position: right 0.5em center;
  background-size: 8px 6px;
  background-repeat: no-repeat;
}
.bk-root select[multiple].bk-input,
.bk-root select[size].bk-input,
.bk-root textarea.bk-input {
  height: auto;
}
.bk-root .bk-input-group {
  width: 100%;
  height: 100%;
  display: inline-flex;
  display: -webkit-inline-flex;
  flex-wrap: nowrap;
  -webkit-flex-wrap: nowrap;
  align-items: start;
  -webkit-align-items: start;
  flex-direction: column;
  -webkit-flex-direction: column;
  white-space: nowrap;
}
.bk-root .bk-input-group.bk-inline {
  flex-direction: row;
  -webkit-flex-direction: row;
}
.bk-root .bk-input-group.bk-inline > *:not(:first-child) {
  margin-left: 5px;
}
.bk-root .bk-input-group input[type="checkbox"] + span,
.bk-root .bk-input-group input[type="radio"] + span {
  position: relative;
  top: -2px;
  margin-left: 3px;
}
`;
export default css;
