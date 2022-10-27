export class Store {
    static uid = "";
    static _instance = null;
    constructor() {
        if (!Store._instance) {
            Store._instance = this;
        }
        return Store._instance;
    }
    static getInstance() {
        return this._instance;
    }

    static languageCode = "en-IN";
}