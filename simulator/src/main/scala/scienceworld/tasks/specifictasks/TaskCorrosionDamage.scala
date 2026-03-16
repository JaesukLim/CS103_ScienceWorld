package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.containers.GlassCup
import scienceworld.objects.electricalcomponent.{Battery, Wire}
import scienceworld.objects.substance.{IronBlock, SaltWater, SodiumChloride}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMakeIsolatedRoom, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalObjectsInSingleContainer}

import scala.collection.mutable.ArrayBuffer

class TaskCorrosionDamage(val mode:String = TaskCorrosionDamage.MODE_CORROSION_MINI) extends TaskParametric {
  val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")
  override val isVisibleInTaskList:Boolean = mode != TaskCorrosionDamage.MODE_CORROSION_UNSEEN

  private val miniRoomName = "test lab"
  private val sourceRooms = Array("living room", "bedroom")
  private val corrodableTargets = Array(
    ("iron block", "rust"),
    ("wire", "corroded wire"),
    ("battery", "corroded battery")
  )

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  if (mode == TaskCorrosionDamage.MODE_CORROSION_MINI) {
    for ((sourceName, targetName) <- corrodableTargets) {
      combinations.append(Array(
        new TaskMakeIsolatedRoom(miniRoomName),
        new TaskObject(sourceName, Some(mkCorrodableObject(sourceName)), miniRoomName, Array.empty[String], forceAdd = true),
        new TaskObject("glass cup", Some(mkSaltWaterCup()), miniRoomName, Array.empty[String], forceAdd = true),
        new TaskValueStr("sourceName", sourceName),
        new TaskValueStr("targetName", targetName),
        new TaskValueBool("isMini", true),
        new TaskValueBool("requiresMix", false)
      ))
    }
  } else if (mode == TaskCorrosionDamage.MODE_CORROSION_SEEN) {
    for ((sourceName, targetName) <- corrodableTargets; roomName <- sourceRooms) {
      combinations.append(Array(
        new TaskObject(sourceName, Some(mkCorrodableObject(sourceName)), roomName, Array.empty[String], forceAdd = true),
        new TaskObject("glass cup", Some(mkSaltWaterCup()), "kitchen", Array.empty[String], forceAdd = true),
        new TaskValueStr("sourceName", sourceName),
        new TaskValueStr("targetName", targetName),
        new TaskValueBool("isMini", false),
        new TaskValueBool("requiresMix", false)
      ))
    }
  } else {
    for ((sourceName, targetName) <- corrodableTargets; roomName <- sourceRooms; requiresMix <- Array(false, true)) {
      val mods = new ArrayBuffer[TaskModifier]()
      mods.append(new TaskObject(sourceName, Some(mkCorrodableObject(sourceName)), roomName, Array.empty[String], forceAdd = true))
      if (requiresMix) {
        mods.append(new TaskObject("glass cup", Some(new GlassCup()), "kitchen", Array.empty[String], forceAdd = true))
        mods.append(new TaskObject("sodium chloride", Some(new SodiumChloride()), "kitchen", Array.empty[String], forceAdd = true))
      } else {
        mods.append(new TaskObject("glass cup", Some(mkSaltWaterCup()), "kitchen", Array.empty[String], forceAdd = true))
      }
      mods.append(new TaskValueStr("sourceName", sourceName))
      mods.append(new TaskValueStr("targetName", targetName))
      mods.append(new TaskValueBool("isMini", false))
      mods.append(new TaskValueBool("requiresMix", requiresMix))
      combinations.append(mods.toArray)
    }
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def mkCorrodableObject(sourceName:String): EnvObject = {
    sourceName match {
      case "iron block" => new IronBlock()
      case "wire" => new Wire()
      case "battery" => new Battery()
      case _ => new IronBlock()
    }
  }

  private def mkSaltWaterCup(): EnvObject = {
    val cup = new GlassCup()
    val saltWater = new SaltWater()
    saltWater.propMaterial.get.temperatureC = 25.0
    saltWater.propMaterial.get.stateOfMatter = "liquid"
    cup.addObject(saltWater)
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

  private def mkDescription(sourceName:String, targetName:String, isMini:Boolean, requiresMix:Boolean): String = {
    if (isMini) {
      "Your task is to corrode the " + sourceName + " until it becomes " + targetName + ". " +
        "Place it into the cup of salt water, wait until the change happens, then focus on the " + targetName + "."
    } else if (requiresMix) {
      "Your task is to damage the " + sourceName + " using corrosion until it becomes " + targetName + ". " +
        "You may need to prepare salt water in a container before the damage can happen. When you are done, focus on the " + targetName + "."
    } else {
      "Your task is to corrode the " + sourceName + " until it becomes " + targetName + ". " +
        "Salt water has been prepared somewhere in the environment. Use it to damage the object, then focus on the " + targetName + "."
    }
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val sourceName = this.getTaskValueStr(modifiers, "sourceName").get
    val targetName = this.getTaskValueStr(modifiers, "targetName").get
    val isMini = this.getTaskValueBool(modifiers, "isMini").getOrElse(false)
    val requiresMix = this.getTaskValueBool(modifiers, "requiresMix").getOrElse(false)

    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalFind(objectName = targetName, failIfWrong = true, description = "focus on the " + targetName))

    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = sourceName, _isOptional = true, description = "be in same location as " + sourceName))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "salt water", _isOptional = true, description = "be in same location as salt water"))
    gSequenceUnordered.append(new GoalObjectsInSingleContainer(objectNames = Array(sourceName, "salt water"), _isOptional = true, description = "place the object in salt water"))
    if (!isMini) {
      gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))
    }

    val description = mkDescription(sourceName, targetName, isMini, requiresMix)
    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = this.setupGoals(this.getCombination(combinationNum), combinationNum)

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array.empty[String])
  }
}

object TaskCorrosionDamage {
  val MODE_CORROSION_MINI = "corrosion mini"
  val MODE_CORROSION_SEEN = "corrosion seen"
  val MODE_CORROSION_UNSEEN = "corrosion unseen"

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskCorrosionDamage(mode = MODE_CORROSION_MINI))
    taskMaker.addTask(new TaskCorrosionDamage(mode = MODE_CORROSION_SEEN))
    taskMaker.addTask(new TaskCorrosionDamage(mode = MODE_CORROSION_UNSEEN))
  }
}
