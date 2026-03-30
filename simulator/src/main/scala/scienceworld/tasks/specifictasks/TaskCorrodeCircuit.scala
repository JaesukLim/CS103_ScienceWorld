package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.containers.GlassCup
import scienceworld.objects.devices.Sink
import scienceworld.objects.electricalcomponent.{Battery, LightBulb, Terminal, Wire}
import scienceworld.objects.substance.{SaltWater, SodiumChloride}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMakeIsolatedRoom, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalActivateDevice, GoalDeactivateDevice, GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation}

import scala.collection.mutable.ArrayBuffer

class TaskCorrodeCircuit(val mode:String = TaskCorrodeCircuit.MODE_CORRODE_CIRCUIT_TINY) extends TaskParametric {
  import TaskCorrodeCircuit._

  override val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")
  override val isVisibleInTaskList:Boolean = mode != TaskCorrodeCircuit.MODE_CORRODE_CIRCUIT_UNSEEN

  private val miniRoomName = "test lab"
  private val circuitRooms = Array("workshop", "art studio")
  private val supplyRooms = Array("living room", "bedroom")

  private val targetSpecs = Array(
    CircuitTarget("wire", "corroded wire"),
    CircuitTarget("battery", "corroded battery")
  )

  private val lightBulbName = "signal light bulb"
  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  if (mode == TaskCorrodeCircuit.MODE_CORRODE_CIRCUIT_TINY) {
    for (targetSpec <- targetSpecs) {
      combinations.append(makeCircuitModifiers(
        targetSpec = targetSpec,
        circuitRoom = miniRoomName,
        supplyRoom = miniRoomName,
        isMini = true,
        requiresMix = false
      ))
    }
  } else if (mode == TaskCorrodeCircuit.MODE_CORRODE_CIRCUIT_SEEN) {
    for (targetSpec <- targetSpecs; circuitRoom <- circuitRooms; supplyRoom <- supplyRooms) {
      combinations.append(makeCircuitModifiers(
        targetSpec = targetSpec,
        circuitRoom = circuitRoom,
        supplyRoom = supplyRoom,
        isMini = false,
        requiresMix = false
      ))
    }
  } else {
    for (targetSpec <- targetSpecs; circuitRoom <- circuitRooms; supplyRoom <- supplyRooms) {
      combinations.append(makeCircuitModifiers(
        targetSpec = targetSpec,
        circuitRoom = circuitRoom,
        supplyRoom = supplyRoom,
        isMini = false,
        requiresMix = true
      ))
    }
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def makeCircuitModifiers(
    targetSpec:CircuitTarget,
    circuitRoom:String,
    supplyRoom:String,
    isMini:Boolean,
    requiresMix:Boolean
  ): Array[TaskModifier] = {
    val modifiers = new ArrayBuffer[TaskModifier]()

    if (isMini) {
      modifiers.append(new TaskMakeIsolatedRoom(miniRoomName))
    }

    modifiers.append(new TaskObject("sink", Some(new Sink(None)), circuitRoom, Array.empty[String], forceAdd = true))
    modifiers.append(new TaskObject(lightBulbName, Some(new LightBulb("signal")), circuitRoom, Array.empty[String], forceAdd = true))

    if (targetSpec.sourceName == "wire") {
      modifiers.append(new TaskObject("battery", Some(new Battery()), circuitRoom, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("red wire", Some(mkNamedWire("red wire")), circuitRoom, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("wire", Some(new Wire()), circuitRoom, Array("sink"), forceAdd = true))
    } else {
      modifiers.append(new TaskObject("battery", Some(new Battery()), circuitRoom, Array("sink"), forceAdd = true))
      modifiers.append(new TaskObject("red wire", Some(mkNamedWire("red wire")), circuitRoom, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("blue wire", Some(mkNamedWire("blue wire")), circuitRoom, Array.empty[String], forceAdd = true))
    }

    if (requiresMix) {
      modifiers.append(new TaskObject("glass cup", Some(new GlassCup()), supplyRoom, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("sodium chloride", Some(new SodiumChloride()), supplyRoom, Array.empty[String], forceAdd = true))
    } else {
      modifiers.append(new TaskObject("glass cup", Some(mkSaltWaterCup()), supplyRoom, Array.empty[String], forceAdd = true))
    }

    modifiers.append(new TaskConfigureCorrosionCircuit(circuitRoom = circuitRoom, lightBulbName = lightBulbName, targetSourceName = targetSpec.sourceName))
    modifiers.append(new TaskValueStr("sourceName", targetSpec.sourceName))
    modifiers.append(new TaskValueStr("targetName", targetSpec.targetName))
    modifiers.append(new TaskValueStr("circuitRoom", circuitRoom))
    modifiers.append(new TaskValueStr("supplyRoom", supplyRoom))
    modifiers.append(new TaskValueStr("lightBulbName", lightBulbName))
    modifiers.append(new TaskValueBool("isMini", isMini))
    modifiers.append(new TaskValueBool("requiresMix", requiresMix))

    modifiers.toArray
  }

  private def mkNamedWire(name:String): Wire = {
    val wire = new Wire()
    wire.name = name
    wire
  }

  private def mkSaltWaterCup(): EnvObject = {
    val cup = new GlassCup()
    cup.addObject(new SaltWater())
    cup
  }

  private def runModifiers(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) return (false, "ERROR: Error running one or more modifiers while setting up task environment.")
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent): (Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (false, "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ").")
    }
    this.runModifiers(this.getCombination(combinationNum), universe, agent)
  }

  private def mkDescription(isMini:Boolean, requiresMix:Boolean, targetName:String): String = {
    if (isMini) {
      "Your task is to use corrosion to break the powered circuit so the light bulb turns off. Salt water is already available in this room. Damage the target component until it becomes " + targetName + ", then focus on the light bulb and the " + targetName + "."
    } else if (requiresMix) {
      "Your task is to break the powered circuit by corrosion. You will need to prepare salt water before you can damage the target component. When the light bulb turns off, focus on it and then on the damaged component."
    } else {
      "Your task is to break the powered circuit by corrosion. Salt water has already been prepared somewhere in the environment. When the light bulb turns off, focus on it and then on the damaged component."
    }
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val sourceName = this.getTaskValueStr(modifiers, "sourceName").get
    val targetName = this.getTaskValueStr(modifiers, "targetName").get
    val lightBulbName = this.getTaskValueStr(modifiers, "lightBulbName").get
    val isMini = this.getTaskValueBool(modifiers, "isMini").getOrElse(false)
    val requiresMix = this.getTaskValueBool(modifiers, "requiresMix").getOrElse(false)

    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalDeactivateDevice(deviceName = lightBulbName, description = "focus on the " + lightBulbName + " after the circuit breaks"))
    gSequence.append(new GoalFind(objectName = targetName, failIfWrong = true, description = "focus on the " + targetName))

    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalActivateDevice(deviceName = lightBulbName, _isOptional = true, description = "focus on the powered " + lightBulbName))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = sourceName, _isOptional = true, description = "find the target component"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "glass cup", _isOptional = true, description = "find the cup"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "salt water", _isOptional = true, description = "find salt water"))
    if (requiresMix) {
      gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "sodium chloride", _isOptional = true, description = "find the sodium chloride"))
    }
    if (!isMini) {
      gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))
    }

    val description = mkDescription(isMini, requiresMix, targetName)
    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = this.setupGoals(this.getCombination(combinationNum), combinationNum)

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array.empty[String])
  }
}

class TaskConfigureCorrosionCircuit(val circuitRoom:String, val lightBulbName:String, val targetSourceName:String) extends TaskModifier {
  override def runModifier(universe:EnvObject, agent:Agent): Boolean = {
    val roomOpt = universe.getContainedObjectsAndPortalsRecursive()
      .collectFirst { case obj if obj.name == circuitRoom => obj }

    if (roomOpt.isEmpty) return false
    val room = roomOpt.get

    def findFirst[A](predicate:PartialFunction[EnvObject, A]): Option[A] = {
      room.getContainedObjectsRecursive().collectFirst(predicate)
    }

    def connect(termA:Terminal, termB:Terminal): Boolean = {
      val okA = termA.propElectricalConnection.get.addConnection(termB)
      val okB = termB.propElectricalConnection.get.addConnection(termA)
      okA && okB
    }

    val lightBulbOpt = findFirst { case bulb:LightBulb if bulb.name == lightBulbName => bulb }
    if (lightBulbOpt.isEmpty) return false
    val lightBulb = lightBulbOpt.get

    if (targetSourceName == "wire") {
      val batteryOpt = findFirst { case battery:Battery if battery.name == "battery" => battery }
      val helperWireOpt = findFirst { case wire:Wire if wire.name == "red wire" => wire }
      val targetWireOpt = findFirst { case wire:Wire if wire.name == "wire" => wire }

      if (batteryOpt.isEmpty || helperWireOpt.isEmpty || targetWireOpt.isEmpty) return false

      val battery = batteryOpt.get
      val helperWire = helperWireOpt.get
      val targetWire = targetWireOpt.get

      val success =
        connect(battery.anode, helperWire.terminal1.get) &&
        connect(helperWire.terminal2.get, lightBulb.cathode) &&
        connect(lightBulb.anode, targetWire.terminal1.get) &&
        connect(targetWire.terminal2.get, battery.cathode)

      if (!success) return false
    } else {
      val batteryOpt = findFirst { case battery:Battery if battery.name == "battery" => battery }
      val wireAOpt = findFirst { case wire:Wire if wire.name == "red wire" => wire }
      val wireBOpt = findFirst { case wire:Wire if wire.name == "blue wire" => wire }

      if (batteryOpt.isEmpty || wireAOpt.isEmpty || wireBOpt.isEmpty) return false

      val battery = batteryOpt.get
      val wireA = wireAOpt.get
      val wireB = wireBOpt.get

      val success =
        connect(battery.anode, wireA.terminal1.get) &&
        connect(wireA.terminal2.get, lightBulb.cathode) &&
        connect(lightBulb.anode, wireB.terminal1.get) &&
        connect(wireB.terminal2.get, battery.cathode)

      if (!success) return false
    }

    if (lightBulb.propDevice.isDefined) {
      lightBulb.propDevice.get.isActivated = true
    }

    true
  }
}

object TaskCorrodeCircuit {
  val MODE_CORRODE_CIRCUIT_TINY = "corrode circuit tiny"
  val MODE_CORRODE_CIRCUIT_SEEN = "corrode circuit seen"
  val MODE_CORRODE_CIRCUIT_UNSEEN = "corrode circuit unseen"

  case class CircuitTarget(sourceName:String, targetName:String)

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskCorrodeCircuit(mode = MODE_CORRODE_CIRCUIT_TINY))
    taskMaker.addTask(new TaskCorrodeCircuit(mode = MODE_CORRODE_CIRCUIT_SEEN))
    taskMaker.addTask(new TaskCorrodeCircuit(mode = MODE_CORRODE_CIRCUIT_UNSEEN))
  }
}
