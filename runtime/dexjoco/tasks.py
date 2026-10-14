SINGLE_ARM_TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
BIMANUAL_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
)

DEFAULT_TASK_PROMPTS = {
    "bimanual_assembly": "Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.",
    "bimanual_hanoi": "Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand.",
    "bimanual_microwave_cook": "Open the microwave door, place the food inside the microwave, close the door, and press the start button.",
    "bimanual_photograph": "Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand.",
    "bimanual_unlock_ipad": "Grasp the iPad and enter the password 123 to unlock the device.",
    "click_mouse": "Move the mouse to the purple mouse pad and click the left mouse button.",
    "fold_glasses": "Fold the glasses and place them into the case.",
    "hammer_nail": "Use the hammer to drive the nail into the wooden board.",
    "pick_bucket": "Place the boxed food into the bucket and then lift the bucket.",
    "pinch_tongs": "Grasp the tongs and perform three consecutive open-close motions.",
    "water_plant": "Grasp the watering can and apply water to the plant.",
}

RAW_PAD_NAMES = (
    "LF_BASE", "TH_TIP", "PALM_1", "LF_TIP", "FF_PAD",
    "FF_TIP", "RF_PAD", "FF_BASE", "MF_TIP", "MF_PAD",
    "RF_TIP", "TH_PAD", "MF_BASE", "RF_BASE", "LF_PAD",
)
