var __asyncValues = (this && this.__asyncValues) || function (o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
};
import { Document } from "../document";
import { Receiver } from "../protocol/receiver";
import { logger } from "../core/logging";
import { stylesheet } from "../core/dom";
import { size, values } from "../core/util/object";
import { add_document_standalone } from "./standalone";
import { _resolve_element, _resolve_root_elements } from "./dom";
import root_css from "../styles/root.css";
import logo_css from "../styles/logo.css";
import notebook_css from "../styles/notebook.css";
stylesheet.append(root_css);
stylesheet.append(logo_css);
stylesheet.append(notebook_css);
// This exists to allow the @bokeh/jupyter_bokeh extension to store the
// notebook kernel so that _init_comms can register the comms target.
// This has to be available at Bokeh.embed.kernels in JupyterLab.
export const kernels = {};
function _handle_notebook_comms(receiver, comm_msg) {
    if (comm_msg.buffers.length > 0)
        receiver.consume(comm_msg.buffers[0].buffer);
    else
        receiver.consume(comm_msg.content.data);
    const msg = receiver.message;
    if (msg != null)
        this.apply_json_patch(msg.content, msg.buffers);
}
function _init_comms(target, doc) {
    if (typeof Jupyter !== 'undefined' && Jupyter.notebook.kernel != null) {
        logger.info(`Registering Jupyter comms for target ${target}`);
        const comm_manager = Jupyter.notebook.kernel.comm_manager;
        try {
            comm_manager.register_target(target, (comm) => {
                logger.info(`Registering Jupyter comms for target ${target}`);
                const r = new Receiver();
                comm.on_msg(_handle_notebook_comms.bind(doc, r));
            });
        }
        catch (e) {
            logger.warn(`Jupyter comms failed to register. push_notebook() will not function. (exception reported: ${e})`);
        }
    }
    else if (doc.roots()[0].id in kernels) {
        logger.info(`Registering JupyterLab comms for target ${target}`);
        const kernel = kernels[doc.roots()[0].id];
        try {
            kernel.registerCommTarget(target, (comm) => {
                logger.info(`Registering JupyterLab comms for target ${target}`);
                const r = new Receiver();
                comm.onMsg = _handle_notebook_comms.bind(doc, r);
            });
        }
        catch (e) {
            logger.warn(`Jupyter comms failed to register. push_notebook() will not function. (exception reported: ${e})`);
        }
    }
    else if (typeof google != 'undefined' && google.colab.kernel != null) {
        logger.info(`Registering Google Colab comms for target ${target}`);
        const comm_manager = google.colab.kernel.comms;
        try {
            comm_manager.registerTarget(target, async (comm) => {
                var e_1, _a;
                var _b;
                logger.info(`Registering Google Colab comms for target ${target}`);
                const r = new Receiver();
                try {
                    for (var _c = __asyncValues(comm.messages), _d; _d = await _c.next(), !_d.done;) {
                        const message = _d.value;
                        const content = { data: message.data };
                        const buffers = [];
                        for (const buffer of (_b = message.buffers) !== null && _b !== void 0 ? _b : []) {
                            buffers.push(new DataView(buffer));
                        }
                        const msg = { content, buffers };
                        _handle_notebook_comms.bind(doc)(r, msg);
                    }
                }
                catch (e_1_1) { e_1 = { error: e_1_1 }; }
                finally {
                    try {
                        if (_d && !_d.done && (_a = _c.return)) await _a.call(_c);
                    }
                    finally { if (e_1) throw e_1.error; }
                }
            });
        }
        catch (e) {
            logger.warn(`Google Colab comms failed to register. push_notebook() will not function. (exception reported: ${e})`);
        }
    }
    else {
        console.warn(`Jupyter notebooks comms not available. push_notebook() will not function. If running JupyterLab ensure the latest @bokeh/jupyter_bokeh extension is installed. In an exported notebook this warning is expected.`);
    }
}
export function embed_items_notebook(docs_json, render_items) {
    if (size(docs_json) != 1)
        throw new Error("embed_items_notebook expects exactly one document in docs_json");
    const document = Document.from_json(values(docs_json)[0]);
    for (const item of render_items) {
        if (item.notebook_comms_target != null)
            _init_comms(item.notebook_comms_target, document);
        const element = _resolve_element(item);
        const roots = _resolve_root_elements(item);
        add_document_standalone(document, element, roots);
    }
}
//# sourceMappingURL=notebook.js.map