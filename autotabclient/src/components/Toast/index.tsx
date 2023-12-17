import toast from "react-hot-toast";

export function HotToastSucess(text: string) {
    toast(text, {

        style: {
            borderRadius: '10px',
            background: '#478572',
            color: '#fff',
        },
    })
}

export function HotToastMessage(text: string) {
    toast(text, {
        style: {
            borderRadius: '10px',
            background: '#333',
            color: '#fff',
        }
    })
}
export function HotToastError(text: string) {
    toast(text, {
        style: {
            borderRadius: '10px',
            background: '#854747',
            color: '#fff',
        }
    })
}
export function HotToastWarning(text: string) {
    toast(text, {
        style: {
            borderRadius: '10px',
            background: '#7d6d2d',
            color: '#fff',
        }
    })
}
